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

    1. harmonic selection (`harmonics`):
         int  n          -> the first n harmonics (index 0 .. n-1)
         list of index   -> only these harmonics (0-based index along the stored H axis,
                            index 0 = 1st harmonic), e.g. [0, 1, 2, 3, 5, 7, 11, 15]
    2. normalization: BatchNorm over every (harmonic, coefficient) pair, because
       EFD magnitudes decay quickly with the harmonic order
    3. 1D CNN along the (selected) harmonic axis (shared by all parts) + harmonic / part embeddings
    4. harmonic self-attention + attention pooling over harmonics
    Parts whose contour is missing (all-zero EFD) are replaced by a learnable token.
    """

    def __init__(self, num_parts=8, harmonics=32, num_coeffs=4, dim=128,
                 attn_layers=1, num_heads=4, dropout=0.1, num_harmonics=None):
        super().__init__()
        if num_harmonics is not None:          # legacy key
            harmonics = num_harmonics
        index = list(range(harmonics)) if isinstance(harmonics, int) else [int(h) for h in harmonics]
        assert len(index) > 0 and min(index) >= 0 and len(set(index)) == len(index), \
            'efd.harmonics must be a positive int or a list of distinct non-negative indices'
        num_harmonics = len(index)
        self.register_buffer('harmonic_index', torch.tensor(index, dtype=torch.long), persistent=False)
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
        max_index = int(self.harmonic_index.max())
        assert P == self.P and Co == self.Co and H_all > max_index, \
            'EFD shape {} incompatible with (P={}, harmonic index up to {}, Co={})'.format(
                tuple(efd.shape), self.P, max_index, self.Co)
        x = efd.index_select(2, self.harmonic_index).reshape(N * P, self.H, Co)
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
    Keypoint coordinates [N, 17, 3] (x, y, conf)
        -> normalized part coordinates [N, P, K, 2], part confidence [N, P, K], edge graph

    * adds virtual joints 17 = neck (mid-shoulder), 18 = mid-hip
    * center-normalization : subtract the mid-hip
    * scale-normalization  : divide by the per-frame body height (y-extent of confident joints)
    * edge graph, two modes (ablation):
        mode='data'  : fixed chain topology of every part + self loops, D^-1/2 A D^-1/2   -> [P, K, K]
                       (pure coordinate data)
        mode='image' : SkeletonGait heatmap convention - joints placed on a canvas where the body is
                       `canvas_height` px tall and a Gaussian kernel exp(-d^2 / 2 sigma^2) (sigma=8)
                       is added to the chain topology, re-normalized per frame          -> [N, P, K, K]
    * augmentation (training only):
        frame_drop_prob : a whole frame's detection is lost (all confidences -> 0)
        joint_drop_prob : single joints are lost (confidence -> 0)
        joint_noise_std : Gaussian noise on the normalized coordinates (unit = body height)
    No learnable parameters.
    """

    def __init__(self, parts=None, conf_thresh=0.1, mode='data', edge_sigma=8.0, canvas_height=128.0,
                 frame_drop_prob=0., joint_drop_prob=0., joint_noise_std=0.):
        super().__init__()
        self.frame_drop_prob, self.joint_drop_prob = frame_drop_prob, joint_drop_prob
        self.joint_noise_std = joint_noise_std
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

        if self.training and self.frame_drop_prob > 0:
            conf = conf * (torch.rand(N, 1, device=conf.device) >= self.frame_drop_prob).to(conf.dtype)
        if self.training and self.joint_drop_prob > 0:
            conf = conf * (torch.rand(N, J, device=conf.device) >= self.joint_drop_prob).to(conf.dtype)

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
        if self.training and self.joint_noise_std > 0:
            xy = xy + torch.randn_like(xy) * self.joint_noise_std

        # parsing [N, 19, .] -> [N, P, K, .]
        flat = self.part_index.view(-1)
        p_xy = xy[:, flat].view(N, self.P, self.K, 2)
        p_conf = conf[:, flat].view(N, self.P, self.K)

        if self.mode == 'data':
            return p_xy, p_conf, self.edge_graph
        # mode == 'image': Gaussian kernel on the heatmap canvas
        d = torch.cdist(p_xy.float(), p_xy.float()) * self.canvas_height           # [N, P, K, K] (px)
        gauss = torch.exp(-d.pow(2) / (2 * self.edge_sigma ** 2))
        off_diag = 1. - torch.eye(self.K, device=kpts.device)
        A = self.topology.unsqueeze(0) + gauss * off_diag
        return p_xy, p_conf, self._sym_norm(A).to(kpts.dtype)


def joint_features(p_xy, p_conf, n, s, seqL=None, use_velocity=True):
    """
    Per-joint input features for the GCN.
        position : (x * c, y * c, c)                                   -> 3 channels
        velocity : (x_t - x_{t-1}, y_t - y_{t-1}) * min(c_t, c_{t-1})   -> +2 channels
    p_xy [n*s, P, K, 2], p_conf [n*s, P, K]; frames are ordered as n sequences of s frames
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
