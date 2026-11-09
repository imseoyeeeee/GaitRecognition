"""
Building blocks of FreQait (see configs/freqait/*.yaml for hyper-parameters).

Frame-level pipeline (N = B*T frames, P = 8 parts):

  EFD      [N, P, H, Co] --EFDEncoder-----------> [N, P, D]   (1D CNN + harmonic attention)
  Keypoint [N, 17, 3]    --SkeletonParser-------> parts [N, P, K, 3], edge graph [P, K, K] ('data')
                                                   or [N, P, K, K] ('image', sigma=8 Gaussian)
                         --PartGCNEncoder-------> [N, P, K, D] -> joint attention pool -> [N, P, D]
  AttentionFusion (SkeletonGait++ style)          [N, P, D]
  PartTransformer (parts + CLS token)             [N, C]
"""
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
# Each part is an ordered chain: consecutive joints are connected.
# ---------------------------------------------------------------------------
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


# =============================================================================
# Silhouette (EFD) branch
# =============================================================================
class EFDEncoder(nn.Module):
    """
    EFD [N, P, H_all, Co] -> [N, P, D]

    1. keep the first `num_harmonics` harmonics
    2. normalization: BatchNorm over every (harmonic, coefficient) pair, because
       EFD magnitudes decay quickly with the harmonic order
    3. 1D CNN along the harmonic axis (shared by all parts) + harmonic / part embeddings
    4. harmonic self-attention + attention pooling over harmonics
    Parts whose contour is missing (all-zero EFD) are replaced by a learnable token.
    """

    def __init__(self, num_parts=8, num_harmonics=32, num_coeffs=4, dim=128,
                 attn_layers=1, num_heads=4, dropout=0.1):
        super().__init__()
        self.P, self.H, self.Co, self.D = num_parts, num_harmonics, num_coeffs, dim

        self.norm = nn.BatchNorm1d(num_harmonics * num_coeffs)
        self.cnn = nn.Sequential(
            nn.Conv1d(num_coeffs, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.harmonic_embedding = nn.Parameter(torch.randn(1, num_harmonics, dim) * 0.02)
        self.part_embedding = nn.Parameter(torch.randn(num_parts, 1, dim) * 0.02)
        self.attention = nn.Sequential(*[TransformerBlock(dim, num_heads, dropout) for _ in range(attn_layers)])
        self.pooling = HarmonicAttentionPooling(dim)
        self.missing_token = nn.Parameter(torch.zeros(num_parts, dim))

    def forward(self, efd):
        N, P, H_all, Co = efd.shape
        assert P == self.P and Co == self.Co and H_all >= self.H, \
            'EFD shape {} incompatible with (P={}, H={}, Co={})'.format(tuple(efd.shape), self.P, self.H, self.Co)
        x = efd[:, :, :self.H].reshape(N * P, self.H, Co)
        valid = x.abs().sum(dim=(1, 2)) > 0                               # [NP]

        x = self.norm(x.reshape(N * P, self.H * Co)).reshape(N * P, self.H, Co)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)                    # [NP, H, D]
        x = x + self.harmonic_embedding
        x = (x.view(N, P, self.H, self.D) + self.part_embedding).view(N * P, self.H, self.D)
        x = self.attention(x)                                             # [NP, H, D]
        x = self.pooling(x).view(N, P, self.D)                            # [N, P, D]

        missing = self.missing_token.unsqueeze(0).expand(N, -1, -1).to(x.dtype)
        return torch.where(valid.view(N, P, 1), x, missing)


# =============================================================================
# Skeleton branch
# =============================================================================
class SkeletonParser(nn.Module):
    """
    Keypoint coordinates [N, 17, 3] (x, y, conf) -> parts [N, P, K, 3], edge graph [P, K, K], conf [N, P, K]

    * adds virtual joints 17 = neck (mid-shoulder), 18 = mid-hip
    * center-normalization : subtract the mid-hip
    * scale-normalization  : divide by the per-frame body height (y-extent of confident joints)
    * edge graph, two modes (ablation):
        mode='data'  : fixed chain topology of every part + self loops, D^-1/2 A D^-1/2   -> [P, K, K]
                       (pure coordinate data)
        mode='image' : SkeletonGait heatmap convention - joints placed on a canvas where the body is
                       `canvas_height` px tall and a Gaussian kernel exp(-d^2 / 2 sigma^2) (sigma=8)
                       is added to the chain topology, re-normalized per frame          -> [N, P, K, K]
    No learnable parameters.
    """

    def __init__(self, parts=None, conf_thresh=0.1, mode='data', edge_sigma=8.0, canvas_height=128.0):
        super().__init__()
        parts = DEFAULT_PARTS if parts is None else parts
        K = len(parts[0])
        assert all(len(p) == K for p in parts), 'every part needs the same number of joints'
        assert mode in ('data', 'image'), "skeleton mode must be 'data' or 'image'"
        self.P, self.K = len(parts), K
        self.conf_thresh = conf_thresh
        self.mode, self.edge_sigma, self.canvas_height = mode, edge_sigma, canvas_height

        self.register_buffer('part_index', torch.tensor(parts, dtype=torch.long))  # [P, K]
        A = torch.eye(K).repeat(self.P, 1, 1)
        idx = torch.arange(K - 1)
        A[:, idx, idx + 1] = 1.
        A[:, idx + 1, idx] = 1.
        self.register_buffer('topology', A)                                        # [P, K, K]
        self.register_buffer('edge_graph', self._sym_norm(A))                      # [P, K, K]

    @staticmethod
    def _sym_norm(A):
        d = A.sum(-1).clamp(min=1e-6).pow(-0.5)
        return d.unsqueeze(-1) * A * d.unsqueeze(-2)

    def forward(self, kpts):
        N, J, _ = kpts.shape
        xy, conf = kpts[..., :2], kpts[..., 2].clamp(0, 1)

        # virtual joints
        neck = (xy[:, 5] + xy[:, 6]) / 2
        midhip = (xy[:, 11] + xy[:, 12]) / 2
        xy = torch.cat([xy, neck[:, None], midhip[:, None]], 1)                      # [N, 19, 2]
        conf = torch.cat([conf,
                          torch.min(conf[:, 5], conf[:, 6])[:, None],
                          torch.min(conf[:, 11], conf[:, 12])[:, None]], 1)          # [N, 19]

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

        # parsing [N, 19, .] -> [N, P, K, .]
        flat = self.part_index.view(-1)
        p_xy = xy[:, flat].view(N, self.P, self.K, 2)
        p_conf = conf[:, flat].view(N, self.P, self.K)
        parts = torch.cat([p_xy * p_conf.unsqueeze(-1), p_conf.unsqueeze(-1)], -1)   # [N, P, K, 3]

        if self.mode == 'data':
            return parts, self.edge_graph, p_conf
        # mode == 'image': Gaussian kernel on the heatmap canvas
        d = torch.cdist(p_xy.float(), p_xy.float()) * self.canvas_height           # [N, P, K, K] (px)
        gauss = torch.exp(-d.pow(2) / (2 * self.edge_sigma ** 2))
        off_diag = 1. - torch.eye(self.K, device=kpts.device)
        A = self.topology.unsqueeze(0) + gauss * off_diag
        return parts, self._sym_norm(A).to(kpts.dtype), p_conf


class PartGraphConv(nn.Module):
    """X' = ReLU(BN(A X W)),  X: [N, P, K, Cin], A: [N, P, K, K] or [P, K, K]"""

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
    parts [N, P, K, 3] + graph -> GCN [N, P, K, D] -> joint attention pooling -> [N, P, D]
    GCN weights are shared by all parts; a learnable joint embedding tells them apart.
    """

    def __init__(self, num_parts=8, num_joints=3, in_channels=3, hidden_dim=64, dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.joint_embedding = nn.Parameter(torch.randn(num_parts, num_joints, hidden_dim) * 0.02)
        self.gcn1 = PartGCNBlock(hidden_dim, hidden_dim, dropout)
        self.gcn2 = PartGCNBlock(hidden_dim, dim, dropout)
        self.joint_pool = PartAttentionPooling(dim)

    def forward(self, parts, A, conf):
        N, P, K, _ = parts.shape
        x = self.input_proj(parts) + self.joint_embedding
        x = self.gcn2(self.gcn1(x, A), A)                                 # [N, P, K, D]
        pooled = self.joint_pool(x.reshape(N * P, K, -1), conf.reshape(N * P, K).float())
        return pooled.view(N, P, -1)


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
