"""
P-DTFPNet: DTFPNet with body parts as graph nodes and a gait-prior + coupling graph.

DTFPNet treats every input channel as one scalar time series (one graph node). In FreQait the
C = P x C/P frame channels are P part vectors (PartwiseProjection concatenates the parts), so here

  * nodes    = the P parts, node features = the C/P channels of each part   x: [B, C/P, P, T]
  * time graph      A_T = A_prior + beta_T * W_T
    frequency graph A_F = A_prior + beta_F1 * W_F + beta_F2 * sigmoid(E_g)
    (all P x P entries; W per sample, W_T and E_g per time group)
      A_prior : gait prior (model_cfg.temporal.p_dtfpnet.part_graph, P x P)
      W_ij    = exp(-(1 - c_ij)^2 / sigma^2),  diagonal 0, beta / sigma learned (softplus)
      c_ij    : coupling of parts i, j measured on k learned per-part signals s_i = W_i x_i
        time branch      : |Pearson correlation| inside each time group (in- or anti-phase motion)
        frequency branch : phase-invariant spectral coherence over the whole clip
                           |sum_f S_i(f) S_j(f)*| / sqrt(sum_f |S_i|^2 sum_f |S_j|^2)  (f > 0, rFFT)
      both averaged over the k signals, in [0, 1].
      E_g     : learned P x P matrix per time group (input independent, as the DTFPNet graphs), diagonal 0,
                initialised so that sigmoid(E) = learned_init (0.1)
  * time / frequency branches, stages, GIN aggregation, dynamic carry between time groups and
    test-time windows are the ones of DTFPNet; the DCT-GRU runs per part over the part features.

Output: [B, out_dim, P] (head part) or [B, out_dim * P, 1] (head global: parts concatenated).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dtfpnet import DTFPNet, DCT_GRU, DynamicGINConv2d

# parts: 0 full body, 1 head, 2 torso, 3 left arm, 4 right arm, 5 left leg, 6 right leg, 7 dress
DEFAULT_PART_GRAPH = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 1],
]
DEFAULT_PART_NAMES = ['full body', 'head', 'torso', 'left arm', 'right arm', 'left leg', 'right leg', 'dress']


def part_names(P):
    return DEFAULT_PART_NAMES if P == len(DEFAULT_PART_NAMES) else ['part {}'.format(i) for i in range(P)]


def inv_softplus(v):
    return math.log(math.expm1(v))


class PartGINConv2d(DynamicGINConv2d):
    """DynamicGINConv2d with a per-sample adjacency.  x [B, C, P, F], adj [B, G, P, P] -> [B, C, P, F]"""

    def forward(self, x, adj, add_loop=True):
        B, C, N, _ = x.size()
        G = adj.size(1)
        adj = self.norm(adj).to(x.dtype)
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)                    # [B, C, G, N, F//G]
        out = torch.einsum('bgnm,bcgmf->bcgnf', adj, x)
        out = torch.cat([out[:, :, :1], out[:, :, 1:] + x[:, :, :-1]], 2)  # dynamic: add previous group
        if add_loop:
            out = (1 + self.eps) * x + out
        return out.transpose(2, 3).reshape(B, C, N, -1)


class CouplingGraph(nn.Module):
    """A = A_prior + beta * exp(-(1 - c)^2 / sigma^2) for one branch ('time' | 'freq')."""

    def __init__(self, prior, part_dim, mode, signal_dim=4, beta=1.0, sigma=0.5, eps=1e-6):
        super().__init__()
        assert mode in ('time', 'freq')
        P = prior.size(0)
        self.mode, self.eps = mode, eps
        self.register_buffer('prior', prior.clone(), persistent=False)
        self.register_buffer('off_diag', 1. - torch.eye(P), persistent=False)
        self.proj = nn.Parameter(nn.init.xavier_uniform_(torch.empty(P, part_dim, signal_dim)))
        self.beta_raw = nn.Parameter(torch.tensor(inv_softplus(beta)))
        self.sigma_raw = nn.Parameter(torch.tensor(inv_softplus(sigma)))

    @property
    def beta(self):
        return F.softplus(self.beta_raw)

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw)

    def coupling(self, x, groups):
        """x [B, C/P, P, L] -> c [B, G, P, P] in [0, 1]"""
        s = torch.einsum('bcpl,pck->bpkl', x.float(), self.proj.float())    # [B, P, k, L]
        B, P, k, L = s.shape
        if self.mode == 'time':
            s = s.view(B, P, k, groups, L // groups)
            s = s - s.mean(-1, keepdim=True)
            cov = torch.einsum('bpkgl,bqkgl->bgkpq', s, s)
            var = cov.diagonal(dim1=-2, dim2=-1)                                # [B, G, k, P]
            c = cov.abs() / torch.sqrt(var.unsqueeze(-1) * var.unsqueeze(-2) + self.eps)
            return c.mean(2)                                                    # [B, G, P, P]
        s = s - s.mean(-1, keepdim=True)
        S = torch.fft.rfft(s, dim=-1)[..., 1:]                                  # [B, P, k, F], no DC
        cross = torch.einsum('bpkf,bqkf->bkpq', S, S.conj())
        power = cross.diagonal(dim1=-2, dim2=-1).real                           # [B, k, P]
        c = cross.abs() / torch.sqrt(power.unsqueeze(-1) * power.unsqueeze(-2) + self.eps)
        return c.mean(1).unsqueeze(1).expand(-1, groups, -1, -1)               # [B, G, P, P]

    def forward(self, x, groups):
        with torch.cuda.amp.autocast(enabled=False):
            c = self.coupling(x, groups).clamp(0., 1.)
            w = torch.exp(-(1. - c) ** 2 / self.sigma ** 2) * self.off_diag
            return self.prior + self.beta * w                                   # [B, G, P, P] fp32


class PDTFPNet(nn.Module):
    """
    Args:
        num_nodes:  C, frame channels (P parts x C/P channels, part-major as from PartwiseProjection)
        num_parts:  P (graph nodes)
        head_part:  True -> [B, out_dim, P];  False -> [B, out_dim * P, 1]
        part_graph: P x P gait prior adjacency (default: DEFAULT_PART_GRAPH)
        signal_dim: k learned coupling signals per part
        beta / sigma: initial values of the learned weight and width of the coupling terms
        coherence:     frequency graph uses the coherence term (beta_F1 * W_F)
        learned_graph: frequency graph uses the learned term (beta_F2 * sigmoid(E_g))
        learned_init:  initial sigmoid(E) of every entry;  beta_learned: initial beta_F2
        others: as DTFPNet
    """
    forward = DTFPNet.forward
    _forward_clips = DTFPNet._forward_clips

    def __init__(self, num_nodes=1024, num_parts=8, head_part=True, part_graph=None, signal_dim=4,
                 beta=1.0, sigma=0.5, coherence=True, learned_graph=True, learned_init=0.1, beta_learned=1.0,
                 in_dim=32, hidden_dim=64, out_dim=128,
                 kern_size=(7, 5, 3), kern_size_mid=(6, 5, 3), groups=2, GRU_layers=1,
                 dropout_size=(0.4, 0.5, 0.5), clip_len=None, clip_stride=15):
        super().__init__()
        assert len(kern_size) == 3 and len(kern_size_mid) == 3 and len(dropout_size) == 3
        assert num_nodes % num_parts == 0, 'part_aggregation.out_dim must be divisible by num_parts'
        P, Cp = num_parts, num_nodes // num_parts
        prior = torch.tensor(DEFAULT_PART_GRAPH if part_graph is None else part_graph, dtype=torch.float32)
        assert prior.shape == (P, P), 'p_dtfpnet.part_graph must be {} x {}'.format(P, P)
        assert (prior >= 0).all(), 'p_dtfpnet.part_graph entries must be >= 0'
        self.P, self.Cp, self.num_graphs = P, Cp, groups
        self.head_part = head_part
        self.clip_len, self.clip_stride = clip_len, clip_stride
        self.out_dim = out_dim if head_part else out_dim * P

        self.register_buffer('prior', prior, persistent=False)
        self.register_buffer('off_diag', 1. - torch.eye(P), persistent=False)
        self.g_time = CouplingGraph(prior, Cp, 'time', signal_dim, beta, sigma)
        self.g_freq = CouplingGraph(prior, Cp, 'freq', signal_dim, beta, sigma) if coherence else None
        if learned_graph:
            assert 0. < learned_init < 1., 'p_dtfpnet.learned_init must be in (0, 1)'
            self.learned_E = nn.Parameter(torch.full((groups, P, P), math.log(learned_init / (1. - learned_init))))
            self.beta_learned_raw = nn.Parameter(torch.tensor(inv_softplus(beta_learned)))
        else:
            self.learned_E = None
        self.asb_in = DCT_GRU(dim=Cp, GRU_layers=GRU_layers)          # per part, over its C/P features
        self.asb_mid = DCT_GRU(dim=in_dim, GRU_layers=GRU_layers)

        self.tconvs_1 = nn.Conv2d(Cp, in_dim, (1, kern_size[0]), padding="same")
        self.gconvs_1 = PartGINConv2d(in_dim, in_dim, groups)
        self.bns_1 = nn.BatchNorm2d(in_dim)
        self.tconvs_tmp_1 = nn.Conv2d(in_dim * 2, in_dim, (1, kern_size_mid[0]), padding="same")
        self.tconvs_tmp_2 = nn.Conv2d(hidden_dim * 2, hidden_dim, (1, kern_size_mid[1]), padding="same")
        self.tconvs_2 = nn.Conv2d(in_dim, hidden_dim, (1, kern_size[1]), padding="same")
        self.gconvs_2 = PartGINConv2d(hidden_dim, hidden_dim, groups)
        self.bns_2 = nn.BatchNorm2d(hidden_dim)
        self.tconvs_3 = nn.Conv2d(hidden_dim, out_dim, (1, kern_size[2]), padding="same")
        self.gconvs_3 = PartGINConv2d(out_dim, out_dim, groups)
        self.bns_3 = nn.BatchNorm2d(out_dim)
        self.dropout_1, self.dropout_2, self.dropout_3 = dropout_size
        self.activation = nn.ReLU()

    def learned_term(self):
        """beta_F2 * sigmoid(E) with zero diagonal  [G, P, P]"""
        return F.softplus(self.beta_learned_raw) * torch.sigmoid(self.learned_E.float()) * self.off_diag

    def freq_graph(self, x):
        """A_F = A_prior (+ beta_F1 * W_F) (+ beta_F2 * sigmoid(E_g))  ->  [B, G, P, P] fp32"""
        if self.g_freq is not None:
            adj = self.g_freq(x, self.num_graphs)
        else:
            adj = self.prior.expand(x.size(0), self.num_graphs, -1, -1)
        if self.learned_E is not None:
            adj = adj + self.learned_term().unsqueeze(0)
        return adj

    @torch.no_grad()
    def report(self):
        """learned graph weights as text lines (for the learned-weights file)"""
        f = lambda v: '{:.4f}'.format(float(v))
        lines = ['Time graph (|correlation| per time segment, groups = {}):'.format(self.num_graphs),
                 '  beta: ' + f(self.g_time.beta), '  sigma: ' + f(self.g_time.sigma)]
        if self.g_freq is not None:
            lines += ['', 'Frequency graph 1 (coherence over the clip):',
                      '  beta: ' + f(self.g_freq.beta), '  sigma: ' + f(self.g_freq.sigma)]
        if self.learned_E is not None:
            names = part_names(self.P)
            lines += ['', 'Frequency graph 2 (learned E):',
                      '  beta: ' + f(F.softplus(self.beta_learned_raw)),
                      '  part order: [{}]'.format(', '.join(names)),
                      '  sigmoid(E)[g][i][j] = weight of part j -> part i in time segment g (diagonal unused, 0)']
            E = (torch.sigmoid(self.learned_E.float()) * self.off_diag).cpu().tolist()
            for g, mat in enumerate(E):
                rows = ',\n'.join('    [' + ', '.join('{:.4f}'.format(v) for v in row) + ']' for row in mat)
                lines += ['  E segment {} = ['.format(g + 1) + '\n' + rows + '\n  ]']
        return lines

    def _gru(self, asb, x):
        """DCT-GRU per part: x [B, C, P, L] -> [B, C, P, L]"""
        B, C, P, L = x.shape
        y = asb(x.permute(0, 2, 1, 3).reshape(B * P, C, L))
        return y.view(B, P, C, L).permute(0, 2, 1, 3)

    def _forward_batch(self, inputs):
        """inputs [B, C, L] -> [B, out_dim, P] | [B, out_dim * P, 1]"""
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            inputs = F.pad(inputs, (int(pad_size), math.ceil(pad_size)), mode='constant', value=0.0)
        B, _, L = inputs.shape
        x = inputs.view(B, self.P, self.Cp, L).transpose(1, 2)            # [B, C/P, P, L]
        adj = self.g_time(x, self.num_graphs)
        adj_fre = self.freq_graph(x)
        fre_asb = self._gru(self.asb_in, x)

        # stage 1: time branch / frequency branch (shared convs), then merge
        x_1 = self.bns_1(self.activation(self.tconvs_1(x)))
        x_1 = self.gconvs_1(x_1, adj)
        x_fre_1 = self.gconvs_1(self.tconvs_1(fre_asb), adj_fre)
        x_1_out = self.tconvs_tmp_1(torch.cat([x_1, x_fre_1], dim=1))
        x_1_out = F.dropout(x_1_out, p=self.dropout_1, training=self.training)

        # stage 2
        x_2 = self.bns_2(self.activation(self.tconvs_2(x_1_out)))
        x_2 = self.gconvs_2(x_2, adj)
        x_fre_2 = self.gconvs_2(self.tconvs_2(self._gru(self.asb_mid, x_fre_1)), adj_fre)
        x_2 = self.tconvs_tmp_2(torch.cat([x_2, x_fre_2], dim=1))
        x_2_out = F.dropout(x_2, p=self.dropout_2, training=self.training)

        # stage 3
        x_3 = self.bns_3(self.activation(self.tconvs_3(x_2_out)))
        x_3 = self.gconvs_3(x_3, adj)
        x_out = F.dropout(x_3, p=self.dropout_3, training=self.training)

        x_out = x_out.mean(-1)                                             # time -> [B, out_dim, P]
        if self.head_part:
            return x_out
        return x_out.reshape(B, -1, 1)                                     # [B, out_dim * P, 1]
