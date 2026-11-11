"""
Building blocks of FreQait (see configs/freqait/*.yaml for hyper-parameters).

Frame-level pipeline (N = B*T frames, P = 8 parts):

  EFD      [N, P, H, Co] --EFDEncoder-----------> [N, P, D]   (1D CNN + harmonic attention)
  Keypoint [N, 17, 3]    --SkeletonParser-------> joint nodes of all parts [N, V, 3], block-diagonal
                                                   graph [V, V] ('data') or [N, V, V] ('image', sigma=8)
                         --PartGCNEncoder-------> [N, V, D] -> joint attention pool per part -> [N, P, D]
  AttentionFusion (SkeletonGait++ style)          [N, P, D]
  PartTransformer (parts + CLS token)             [N, C]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fre_modules import HarmonicSelfAttention, HarmonicAttentionPooling, PartAttentionPooling

# Pre-norm transformer block [N, L, D] -> [N, L, D] (MHA + FFN, residual).
TransformerBlock = HarmonicSelfAttention

# ---------------------------------------------------------------------------
# Default keypoint parsing (COCO-17 + 2 virtual joints), aligned with the EFD
# part order of EFD-Analysis/configs/fineefd.yaml:
#   [full_body(background), head, torso, left_arm, right_arm, left_leg, right_leg, dress]
# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists,
#          11/12 hips, 13/14 knees, 15/16 ankles.   Virtual: 17 neck, 18 mid-hip.
# A part is either a list of joints (ordered chain: consecutive joints are connected)
# or a dict {joints: [...] | all, topology: chain | skeleton} or {joints: [...], edges: [[u, v], ...]}
# (explicit edges, joint ids); parts may have different sizes.
# ---------------------------------------------------------------------------
NUM_JOINTS = 19
DEFAULT_PARTS = [
    [15, 0, 16],   # full body : L-ankle - nose - R-ankle
    [3, 0, 4],     # head      : L-ear - nose - R-ear
    [5, 18, 6],    # torso     : L-shoulder - mid-hip - R-shoulder
    [5, 7, 9],     # left arm
    [6, 8, 10],    # right arm
    [11, 13, 15],  # left leg
    [12, 14, 16],  # right leg
    [11, 17, 12],  # dress     : L-hip - neck - R-hip
]
# human skeleton bones of the 19 joints (a tree), used by `topology: skeleton`
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # face
    (0, 17), (17, 5), (17, 6),                  # nose - neck - shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (17, 18), (18, 11), (18, 12),               # spine, mid-hip - hips
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]


# =============================================================================
# Silhouette (EFD) branch
# =============================================================================
def efd_to_ellipse(coeffs):
    """
    EFD coefficients [..., 4] = (a, b, c, d) of one harmonic,  x = a cos + b sin,  y = c cos + d sin
    -> ellipse parameters [..., 6] = (A, B, e cos 2phi, e sin 2phi, A cos 2theta, A sin 2theta)

    Closed-form 2x2 SVD  [[a, b], [c, d]] = Rot(phi) diag(A, B) Rot(theta):
        A     semi-major axis (>= 0)
        B     semi-minor axis, signed (sign = direction in which the contour is traversed)
        phi   orientation of the major axis (pi-periodic -> 2 phi)   : rotation of the part (limb swing)
        theta phase / start point on the ellipse (-> 2 theta)
    The angles are weighted by how well they are defined: the orientation by the elongation
    e = A - |B| (undefined for a circle) and the phase by A (undefined for a vanishing harmonic),
    so near-circular / tiny harmonics give ~0 instead of noisy unit vectors and the features are
    continuous. Computed in fp32, no learnable parameters.
    """
    a, b, c, d = coeffs.float().unbind(-1)
    E, F_, G, H = (a + d) / 2, (a - d) / 2, (c + b) / 2, (c - b) / 2
    Q, R = torch.sqrt(E ** 2 + H ** 2), torch.sqrt(F_ ** 2 + G ** 2)
    A, B = Q + R, Q - R
    a1, a2 = torch.atan2(G, F_), torch.atan2(H, E)
    two_phi, two_theta = a1 + a2, a2 - a1
    e = A - B.abs()
    out = torch.stack([A, B, e * torch.cos(two_phi), e * torch.sin(two_phi),
                       A * torch.cos(two_theta), A * torch.sin(two_theta)], -1)
    return out.to(coeffs.dtype)


class EFDEncoder(nn.Module):
    """
    EFD [N, P, H_all, Co] -> [N, P, D]

    1. harmonic selection (`harmonics`):
         int  n          -> the first n harmonics (index 0 .. n-1)
         list of index   -> only these harmonics (0-based index along the stored H axis,
                            index 0 = 1st harmonic), e.g. [0, 1, 2, 3, 5, 7, 11, 15]
    2. normalization: BatchNorm over every (harmonic, coefficient) pair, because
       EFD magnitudes decay quickly with the harmonic order
    3. 1D CNN along the (selected) harmonic axis (shared by all parts) + harmonic / part embeddings
    4. harmonic self-attention + attention pooling over harmonics

    Ablation options:
        input   : 'raw'     (a, b, c, d) per harmonic
                  'ellipse' ellipse parameters per harmonic (efd_to_ellipse, 6 features)
        embed   : 'cnn'     1D CNN along the harmonic axis (mixes neighbouring harmonics)
                  'mlp'     the same small MLP on every harmonic (no mixing; attention relates harmonics)
                  (used by encoder 'attention' only)
        encoder : 'attention' embed -> harmonic self-attention -> attention pooling (steps 3-4)
                  'mlp'       one MLP on all (harmonic, feature) values of a part (baseline)
    Parts whose contour is missing (all-zero EFD) are replaced by a learnable token. "Missing" is
    one fixed vector per part, so the appearance / disappearance of a part over time stays a
    pattern for the temporal model.

    Harmonic dropout (training only, augmentation): every selected harmonic is switched off
    with probability `harmonic_drop_prob` by zeroing its normalized coefficients (0 = the mean
    after BatchNorm, i.e. no information). scope 'sequence': the same harmonics are dropped in
    every frame of a sequence (per part), 'frame': independently per frame. keep_first: never
    drop the 1st selected harmonic (the base ellipse: size / orientation of the part).
    """

    def __init__(self, num_parts=8, harmonics=32, num_coeffs=4, dim=128,
                 attn_layers=1, num_heads=4, dropout=0.1, num_harmonics=None,
                 harmonic_drop_prob=0., harmonic_drop_scope='sequence', harmonic_drop_keep_first=True,
                 input='raw', embed='cnn', encoder='attention', mlp_hidden=256):
        super().__init__()
        assert input in ('raw', 'ellipse'), "efd.input must be 'raw' or 'ellipse'"
        assert embed in ('cnn', 'mlp'), "efd.embed must be 'cnn' or 'mlp'"
        assert encoder in ('attention', 'mlp'), "efd.encoder must be 'attention' or 'mlp'"
        assert input == 'raw' or num_coeffs == 4, "efd.input 'ellipse' needs the 4 EFD coefficients (num_coeffs: 4)"
        self.input, self.embed, self.encoder = input, embed, encoder
        assert 0. <= harmonic_drop_prob < 1., 'harmonic_drop_prob must be in [0, 1)'
        assert harmonic_drop_scope in ('sequence', 'frame'), "harmonic_drop_scope must be 'sequence' or 'frame'"
        self.harmonic_drop_prob = harmonic_drop_prob
        self.harmonic_drop_scope = harmonic_drop_scope
        self.harmonic_drop_keep_first = harmonic_drop_keep_first
        if num_harmonics is not None:          # legacy key
            harmonics = num_harmonics
        index = list(range(harmonics)) if isinstance(harmonics, int) else [int(h) for h in harmonics]
        assert len(index) > 0 and min(index) >= 0 and len(set(index)) == len(index), \
            'efd.harmonics must be a positive int or a list of distinct non-negative indices'
        num_harmonics = len(index)
        self.register_buffer('harmonic_index', torch.tensor(index, dtype=torch.long), persistent=False)
        self.P, self.H, self.Co, self.D = num_parts, num_harmonics, num_coeffs, dim
        self.F = 6 if input == 'ellipse' else num_coeffs                   # features per harmonic

        self.norm = nn.BatchNorm1d(num_harmonics * self.F)
        self.missing_token = nn.Parameter(torch.zeros(num_parts, dim))
        if encoder == 'mlp':
            # baseline: one MLP on the flattened [H * F] values of a part (shared by the parts)
            self.mlp_in = nn.Linear(num_harmonics * self.F, mlp_hidden)
            self.mlp_part = nn.Parameter(torch.randn(num_parts, mlp_hidden) * 0.02)
            self.mlp_out = nn.Sequential(
                nn.LayerNorm(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(mlp_hidden, dim))
            return
        if embed == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(self.F, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(96), nn.GELU(),
                nn.Conv1d(96, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(dim), nn.GELU(),
            )
        else:
            # the same MLP on every harmonic token (no mixing of neighbouring harmonics)
            self.harmonic_mlp = nn.Sequential(
                nn.Linear(self.F, 64), nn.GELU(),
                nn.Linear(64, dim), nn.GELU(),
            )
        self.harmonic_embedding = nn.Parameter(torch.randn(1, num_harmonics, dim) * 0.02)
        self.part_embedding = nn.Parameter(torch.randn(num_parts, 1, dim) * 0.02)
        self.attention = nn.Sequential(*[TransformerBlock(dim, num_heads, dropout) for _ in range(attn_layers)])
        self.pooling = HarmonicAttentionPooling(dim)

    def harmonic_dropout(self, x, n):
        """x [N*P, H, F] (normalized), N = n sequences x s frames -> harmonics zeroed at random."""
        if not self.training or self.harmonic_drop_prob <= 0:
            return x
        NP = x.shape[0]
        if self.harmonic_drop_scope == 'sequence' and n is not None:
            s = NP // (n * self.P)
            keep = torch.rand(n, 1, self.P, self.H, device=x.device) >= self.harmonic_drop_prob
            keep = keep.expand(n, s, self.P, self.H).reshape(NP, self.H)
        else:
            keep = torch.rand(NP, self.H, device=x.device) >= self.harmonic_drop_prob
        if self.harmonic_drop_keep_first:
            keep[:, 0] = True
        return x * keep.unsqueeze(-1).to(x.dtype)

    def forward(self, efd, n=None):
        """efd [N, P, H_all, Co]; n = number of sequences in N (for sequence-level harmonic dropout)."""
        N, P, H_all, Co = efd.shape
        max_index = int(self.harmonic_index.max())
        assert P == self.P, \
            'EFD data has {} parts but model_cfg.num_parts is {}. Set num_parts: {} and give skeleton.parts ' \
            '{} joint groups in the same order as the EFD parts (fineefd: 8 parts, coarseefd: 4).'.format(
                P, self.P, P, P)
        assert Co == self.Co and H_all > max_index, \
            'EFD shape {} incompatible with (P={}, harmonic index up to {}, Co={})'.format(
                tuple(efd.shape), self.P, max_index, self.Co)
        x = efd.index_select(2, self.harmonic_index).reshape(N * P, self.H, Co)
        valid = x.abs().sum(dim=(1, 2)) > 0                               # [NP]
        if self.input == 'ellipse':
            x = efd_to_ellipse(x)                                         # [NP, H, 6]

        x = self.norm(x.reshape(N * P, self.H * self.F)).reshape(N * P, self.H, self.F)
        x = self.harmonic_dropout(x, n)
        if self.encoder == 'mlp':
            h = self.mlp_in(x.reshape(N * P, self.H * self.F)).view(N, P, -1) + self.mlp_part
            x = self.mlp_out(h)                                           # [N, P, D]
        else:
            if self.embed == 'cnn':
                x = self.cnn(x.transpose(1, 2)).transpose(1, 2)            # [NP, H, D]
            else:
                x = self.harmonic_mlp(x)                                  # [NP, H, D]
            x = x + self.harmonic_embedding
            x = (x.view(N, P, self.H, self.D) + self.part_embedding).view(N * P, self.H, self.D)
            x = self.attention(x)                                         # [NP, H, D]
            x = self.pooling(x).view(N, P, self.D)                        # [N, P, D]

        missing = self.missing_token.unsqueeze(0).expand(N, -1, -1).to(x.dtype)
        return torch.where(valid.view(N, P, 1), x, missing)


# =============================================================================
# Skeleton branch
# =============================================================================
def parse_part_cfg(parts):
    """-> list of (joints, edges) with edges as (a, b) positions inside the part."""
    out = []
    for i, part in enumerate(parts):
        if isinstance(part, dict):
            joints = part.get('joints', 'all')
            topology = part.get('topology', 'chain')
            explicit = part.get('edges')
        else:
            joints, topology, explicit = part, 'chain', None
        joints = list(range(NUM_JOINTS)) if joints == 'all' else [int(j) for j in joints]
        assert len(joints) > 0 and all(0 <= j < NUM_JOINTS for j in joints), \
            'part {}: joints must be in [0, {}) (17 neck, 18 mid-hip)'.format(i, NUM_JOINTS)
        assert len(set(joints)) == len(joints), 'part {}: duplicated joint'.format(i)
        pos = {j: a for a, j in enumerate(joints)}
        if explicit is not None:                       # edges given as joint ids
            for e in explicit:
                assert len(e) == 2 and e[0] in pos and e[1] in pos, \
                    'part {}: edge {} must connect two joints of the part {}'.format(i, e, joints)
            edges = [(pos[u], pos[v]) for u, v in explicit]
        elif topology == 'chain':
            edges = [(a, a + 1) for a in range(len(joints) - 1)]
        elif topology == 'skeleton':
            edges = [(pos[u], pos[v]) for u, v in SKELETON_EDGES if u in pos and v in pos]
        else:
            raise ValueError("part {}: topology must be 'chain' or 'skeleton'".format(i))
        out.append((joints, edges))
    return out


class SkeletonParser(nn.Module):
    """
    keypoints [N, 17, 3] (x, y, conf)  ->  joint nodes of all parts + edge graph.

    * adds virtual joints 17 = neck (mid-shoulder), 18 = mid-hip, computed on the fly from the 17 COCO
      keypoints (not needed in the data): position and confidence are the mean (= median of two) of
      the two source joints (5, 6 / 11, 12), computed in one step. virtual_require_both=True: the
      confidence is 0 if either source joint is below conf_thresh (its position is unreliable then)
    * center-normalization : subtract the mid-hip
    * scale-normalization  : divide by the per-frame body height (y-extent of confident joints)
    * parsing: the joints of all parts are concatenated into V = sum(|part|) nodes, so parts can
      have different sizes (e.g. full body = all 19 joints, the others 3). The graph is
      block-diagonal (one block per part, parts never mix) and normalized as D^-1/2 A D^-1/2.
    * edge graph, two modes (ablation):
        mode='data'  : fixed topology (chain / skeleton bones) + self loops               -> [V, V]
        mode='image' : SkeletonGait++ heatmap convention - joint distances measured with the body
                       `canvas_height` px tall, a Gaussian exp(-d^2 / 2 sigma^2) (sigma=8) of the
                       distance between two joints of the same part is added:
                       A = topology + alpha_part * Gaussian, re-normalized per frame       -> [N, V, V]
                       edge_conf=True : Gaussian x conf_i * conf_j
                       learn_alpha    : alpha_part learned per part (softplus, init edge_alpha);
                                        otherwise alpha = edge_alpha for every part
    * augmentation (training only):
        frame_drop_prob : a whole frame's detection is lost (all confidences -> 0)
        joint_drop_prob : single joints are lost (confidence -> 0)
        joint_noise_std : Gaussian noise on the normalized coordinates (unit = body height)
    Learnable parameters: only alpha (image mode + learn_alpha).
    """

    def __init__(self, parts=None, conf_thresh=0.1, mode='data', edge_sigma=8.0, canvas_height=128.0,
                 frame_drop_prob=0., joint_drop_prob=0., joint_noise_std=0., edge_conf=False,
                 edge_alpha=1.0, learn_alpha=False, virtual_require_both=True):
        super().__init__()
        self.frame_drop_prob, self.joint_drop_prob = frame_drop_prob, joint_drop_prob
        self.joint_noise_std = joint_noise_std
        assert mode in ('data', 'image'), "skeleton mode must be 'data' or 'image'"
        parsed = parse_part_cfg(DEFAULT_PARTS if parts is None else parts)
        self.P = len(parsed)
        self.part_sizes = [len(j) for j, _ in parsed]
        self.V = sum(self.part_sizes)
        self.conf_thresh = conf_thresh
        self.mode, self.edge_sigma, self.canvas_height = mode, edge_sigma, canvas_height
        self.edge_conf = edge_conf
        self.virtual_require_both = virtual_require_both
        # source joints of the virtual joints 17 (neck) and 18 (mid-hip)
        self.register_buffer('virtual_src', torch.tensor([[5, 6], [11, 12]]), persistent=False)

        index, part_of_node, A = [], [], torch.eye(self.V)
        offset = 0
        for p, (joints, edges) in enumerate(parsed):
            index += joints
            part_of_node += [p] * len(joints)
            for a, b in edges:
                A[offset + a, offset + b] = A[offset + b, offset + a] = 1.
            offset += len(joints)
        part_of_node = torch.tensor(part_of_node, dtype=torch.long)
        same_part = (part_of_node[:, None] == part_of_node[None, :]).float()
        # derived from the config -> not stored in checkpoints
        self.register_buffer('joint_index', torch.tensor(index, dtype=torch.long), persistent=False)   # [V]
        self.register_buffer('part_of_node', part_of_node, persistent=False)                            # [V]
        self.register_buffer('topology', A, persistent=False)                                           # [V, V]
        self.register_buffer('edge_graph', self._sym_norm(A), persistent=False)                         # [V, V]
        self.register_buffer('gauss_mask', same_part * (1. - torch.eye(self.V)), persistent=False)      # [V, V]

        self.learn_alpha = bool(learn_alpha) and mode == 'image'
        if self.learn_alpha:
            assert edge_alpha > 0, 'learn_alpha needs edge_alpha > 0 (initial value, softplus)'
            raw = math.log(math.expm1(edge_alpha))
            self.alpha_raw = nn.Parameter(torch.full((self.P,), raw))
        else:
            self.edge_alpha = float(edge_alpha)

    def alpha(self):
        """[P] weight of the Gaussian term of every part (image mode)."""
        if self.learn_alpha:
            return F.softplus(self.alpha_raw)
        return torch.full((self.P,), self.edge_alpha, device=self.topology.device)

    @staticmethod
    def _sym_norm(A):
        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
        return d.unsqueeze(-1) * A * d.unsqueeze(-2)

    def forward(self, kpts):
        """kpts [N, 17, 3] -> p_xy [N, V, 2], p_conf [N, V], graph [V, V] or [N, V, V]"""
        N, J, _ = kpts.shape
        xy, conf = kpts[..., :2], kpts[..., 2].clamp(0, 1)

        if self.training and self.frame_drop_prob > 0:
            conf = conf * (torch.rand(N, 1, device=conf.device) >= self.frame_drop_prob).to(conf.dtype)
        if self.training and self.joint_drop_prob > 0:
            conf = conf * (torch.rand(N, J, device=conf.device) >= self.joint_drop_prob).to(conf.dtype)

        # virtual joints 17 (neck), 18 (mid-hip): mean of (x, y, conf) of the two source joints in one step
        src = torch.cat([xy, conf.unsqueeze(-1)], -1)[:, self.virtual_src]            # [N, 2, 2, 3]
        virt = src.mean(2)                                                            # [N, 2, 3]
        v_conf = virt[..., 2]
        if self.virtual_require_both:     # a missing source joint makes the midpoint unreliable
            v_conf = v_conf * (src[..., 2].min(-1)[0] >= self.conf_thresh).to(v_conf.dtype)
        xy = torch.cat([xy, virt[..., :2]], 1)                                        # [N, 19, 2]
        conf = torch.cat([conf, v_conf], 1)                                           # [N, 19]

        # center normalization
        xy = xy - xy[:, 18:19]
        # scale normalization by body height
        mask = (conf[:, :J] > self.conf_thresh)
        y = xy[:, :J, 1]
        big = torch.finfo(y.dtype).max
        y_max = torch.where(mask, y, torch.full_like(y, -big)).max(1)[0]
        y_min = torch.where(mask, y, torch.full_like(y, big)).min(1)[0]
        height = y_max - y_min
        height = torch.where((mask.sum(1) >= 2) & (height > 1e-6), height, torch.ones_like(height))
        xy = xy / height.view(N, 1, 1)
        if self.training and self.joint_noise_std > 0:
            xy = xy + torch.randn_like(xy) * self.joint_noise_std

        # parsing [N, 19, .] -> [N, V, .]
        p_xy = xy[:, self.joint_index]
        p_conf = conf[:, self.joint_index]

        if self.mode == 'data':
            return p_xy, p_conf, self.edge_graph
        # mode == 'image': Gaussian kernel on the heatmap canvas, only between joints of the same part
        d = torch.cdist(p_xy.float(), p_xy.float()) * self.canvas_height           # [N, V, V] (px)
        gauss = torch.exp(-d.pow(2) / (2 * self.edge_sigma ** 2)) * self.gauss_mask
        if self.edge_conf:
            c = p_conf.float()
            gauss = gauss * c.unsqueeze(-1) * c.unsqueeze(-2)                          # conf_i * conf_j
        alpha = self.alpha()[self.part_of_node]                                       # [V]
        A = self.topology.unsqueeze(0) + alpha.view(1, -1, 1) * gauss
        return p_xy, p_conf, self._sym_norm(A).to(kpts.dtype)


def joint_features(p_xy, p_conf, n, s, seqL=None, use_velocity=True):
    """
    Per-joint input features for the GCN.
        position : (x * c, y * c, c)                                   -> 3 channels
        velocity : (x_t - x_{t-1}, y_t - y_{t-1}) * min(c_t, c_{t-1})   -> +2 channels
    p_xy [n*s, V, 2], p_conf [n*s, V]; frames are ordered as n sequences of s frames
    (training) or one concatenation of len(seqL) sequences (test, n = 1).
    The first frame of every sequence gets zero velocity.
    """
    c = p_conf.unsqueeze(-1).to(p_xy.dtype)
    feats = [p_xy * c, c]
    if use_velocity:
        xy = p_xy.view(n, s, *p_xy.shape[1:])
        cf = c.view(n, s, *c.shape[1:])
        vel = torch.zeros_like(xy)
        vel[:, 1:] = (xy[:, 1:] - xy[:, :-1]) * torch.min(cf[:, 1:], cf[:, :-1])
        if seqL is not None:                     # no velocity across sequence boundaries
            starts = torch.cumsum(seqL[0].long(), 0)[:-1]
            vel[:, starts] = 0
        feats.append(vel.view_as(p_xy))
    return torch.cat(feats, -1)


class PartGraphConv(nn.Module):
    """X' = ReLU(BN(A X W)),  X: [N, V, Cin], A: [N, V, V] or [V, V] (block-diagonal over parts)"""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, A):
        x = self.fc(torch.matmul(A, x))
        shape = x.shape
        x = self.bn(x.reshape(-1, shape[-1])).view(shape)
        return self.dropout(F.relu(x))


class PartGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.gcn1 = PartGraphConv(in_channels, out_channels, dropout)
        self.gcn2 = PartGraphConv(out_channels, out_channels, dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, A):
        return F.relu(self.gcn2(self.gcn1(x, A), A) + self.residual(x))


class PartGCNEncoder(nn.Module):
    """
    joint nodes [N, V, Cin] + block-diagonal graph -> GCN [N, V, D] -> joint attention pooling
    inside every part -> [N, P, D]. Parts may have different numbers of joints.
    GCN weights are shared by all parts; a learnable node embedding tells the joints apart.
    """

    def __init__(self, part_of_node, in_channels=3, hidden_dim=64, dim=128, dropout=0.1):
        super().__init__()
        part_of_node = torch.as_tensor(part_of_node, dtype=torch.long)
        self.P, self.V = int(part_of_node.max()) + 1, len(part_of_node)
        member = part_of_node[None, :] == torch.arange(self.P)[:, None]               # [P, V]
        self.register_buffer('member', member, persistent=False)
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.joint_embedding = nn.Parameter(torch.randn(self.V, hidden_dim) * 0.02)
        self.gcn1 = PartGCNBlock(hidden_dim, hidden_dim, dropout)
        self.gcn2 = PartGCNBlock(hidden_dim, dim, dropout)
        self.joint_pool = PartAttentionPooling(dim)

    def forward(self, nodes, A, conf):
        x = self.input_proj(nodes) + self.joint_embedding
        x = self.gcn2(self.gcn1(x, A), A)                                              # [N, V, D]
        # attention pooling (PartAttentionPooling scores) restricted to the joints of each part
        scores = self.joint_pool.attention(x).squeeze(-1)                              # [N, V]
        scores = scores + torch.log(conf.float().clamp(min=1e-6)).to(scores.dtype)
        scores = scores.unsqueeze(1).masked_fill(~self.member, float('-inf'))          # [N, P, V]
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum('npv,nvd->npd', weights, x)                                # [N, P, D]


# =============================================================================
# Fusion + frame-level transformer
# =============================================================================
class AttentionFusion(nn.Module):
    """
    SkeletonGait++ attention fusion, part-wise:
    score = MLP([efd, skel]) -> softmax over the 2 modalities (per channel) -> weighted sum.
    efd_feat, skel_feat: [N, P, D] -> [N, P, D]
    """

    def __init__(self, dim=128, squeeze_ratio=4, dropout=0.1):
        super().__init__()
        hidden = max(dim // squeeze_ratio, 1)
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim * 2),
        )

    def forward(self, efd_feat, skel_feat):
        N, P, D = efd_feat.shape
        score = self.attention(torch.cat([efd_feat, skel_feat], -1)).view(N, P, 2, D)
        w = F.softmax(score, dim=2)
        return efd_feat * w[:, :, 0] + skel_feat * w[:, :, 1]


class PartwiseProjection(nn.Module):
    """
    [N, P, D] -> [N, C]  (recommended part aggregation, light)
    Every part gets its own Linear(D -> C/P); the outputs are concatenated, so each
    channel group of the frame feature belongs to one part. Inter-part coordination is
    left to the temporal module (DTFPNet's node graph runs over these channels).
    """

    def __init__(self, num_parts=8, dim=128, out_dim=128):
        super().__init__()
        assert out_dim % num_parts == 0, 'out_dim must be divisible by num_parts'
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_parts, dim, out_dim // num_parts)))
        self.bias = nn.Parameter(torch.zeros(num_parts, out_dim // num_parts))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = torch.einsum('npd,pdc->npc', x, self.weight.to(x.dtype)) + self.bias.to(x.dtype)
        return self.norm(x.reshape(x.size(0), -1))


class PartTransformer(nn.Module):
    """[N, P, D] part tokens + CLS -> transformer encoder -> CLS projected to [N, C]"""

    def __init__(self, num_parts=8, dim=128, out_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_parts + 1, dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x):
        N = x.size(0)
        x = torch.cat([self.cls_token.expand(N, -1, -1).to(x.dtype), x], 1) + self.pos_embedding
        x = self.blocks(x)
        return self.proj(self.norm(x[:, 0]))


# =============================================================================
# Temporal transformer (ablation alternative to DTFPNet)
# =============================================================================
class TemporalTransformer(nn.Module):
    """
    Frame features [n, C, s] -> sequence embedding [n, out_dim, 1]  (same interface as DTFPNet)

    Linear(C -> dim) + sinusoidal position encoding -> pre-norm transformer blocks over frames ->
    LayerNorm -> mean over time -> Linear(dim -> out_dim).
    Variable-length test sequences (seqL) are processed one by one, like DTFPNet.

    Train / test length gap: training clips have `clip_len` frames (= frames_num_fixed), while test
    sequences use all frames. In eval mode, a sequence longer than `clip_len` is cut into windows of
    `clip_len` frames (step `clip_stride`, the last window aligned to the end); every window is
    embedded on its own and the embeddings are averaged, so the model only sees positions it was
    trained on. `clip_len: null` disables this (whole sequence at once).
    """

    def __init__(self, num_nodes=128, dim=128, out_dim=128, num_layers=2, num_heads=4, dropout=0.1,
                 clip_len=30, clip_stride=15, num_parts=None):
        super().__init__()
        if num_parts:
            raise ValueError("head.type 'part' is not supported with temporal.type 'transformer': its tokens "
                             "are whole frames, so there is no part axis left. Use head.type: global.")
        self.dim, self.out_dim = dim, out_dim
        self.clip_len, self.clip_stride = clip_len, clip_stride
        self.in_proj = nn.Linear(num_nodes, dim)
        self.blocks = nn.Sequential(*[TransformerBlock(dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def positional_encoding(self, length, device, dtype):
        pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                        * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(length, self.dim, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype).unsqueeze(0)

    def _forward_batch(self, x):
        x = self.in_proj(x.transpose(1, 2))                      # [n, s, dim]
        x = x + self.positional_encoding(x.size(1), x.device, x.dtype)
        x = self.norm(self.blocks(x)).mean(1)                    # [n, dim]
        return self.out_proj(x).unsqueeze(-1)                   # [n, out_dim, 1]

    def _forward_clips(self, x):
        """eval only: x [n, C, L] -> mean of window embeddings [n, out_dim, 1]"""
        n, C, L = x.shape
        if self.training or not self.clip_len or L <= self.clip_len:
            return self._forward_batch(x)
        starts = list(range(0, L - self.clip_len + 1, self.clip_stride))
        if starts[-1] != L - self.clip_len:
            starts.append(L - self.clip_len)                    # cover the tail
        clips = torch.stack([x[:, :, s:s + self.clip_len] for s in starts], 1)   # [n, k, C, clip]
        emb = self._forward_batch(clips.reshape(n * len(starts), C, self.clip_len))
        return emb.view(n, len(starts), *emb.shape[1:]).mean(1)

    def forward(self, x, seqL=None):
        if seqL is None:
            return self._forward_clips(x)
        outs, start = [], 0
        for L in seqL[0].data.cpu().numpy().tolist():
            outs.append(self._forward_clips(x[:, :, start:start + L]))
            start += L
        return torch.cat(outs, 0)


# =============================================================================
# Set pooling (GaitSet-style temporal baseline)
# =============================================================================
class SetPooling(nn.Module):
    """
    Frame features [n, C, s] -> max over frames (order-free, length-agnostic).
        global : [n, C, 1]
        parts  : channels grouped by part (PartwiseProjection order) -> [n, C / P, P]
    No learnable parameters; the head FCs map it to the embedding.
    """

    def __init__(self, num_nodes=64, num_parts=None):
        super().__init__()
        if num_parts:
            assert num_nodes % num_parts == 0, 'num_nodes must be divisible by num_parts'
        self.num_parts = num_parts
        self.out_dim = num_nodes // num_parts if num_parts else num_nodes

    def _pool(self, x):
        x = x.max(-1)[0]                                              # [n, C]
        if self.num_parts:
            return x.view(x.size(0), self.num_parts, -1).transpose(1, 2).contiguous()   # [n, C/P, P]
        return x.unsqueeze(-1)

    def forward(self, x, seqL=None):
        if seqL is None:
            return self._pool(x)
        outs, start = [], 0
        for L in seqL[0].data.cpu().numpy().tolist():
            outs.append(self._pool(x[:, :, start:start + L]))
            start += L
        return torch.cat(outs, 0)
