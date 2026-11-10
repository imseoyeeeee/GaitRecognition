"""
DTFPNet temporal head for FreQait.

Port of `GNNStack` in DTFPNet-DAAC/DTFPNet_temp/main_run.py
(Yu et al., "DTFPNet: Temporal and frequency dynamic graph neural network for
time series classification", Pattern Recognition 2026), adapted for gait:

  * input  : frame-level features as a multivariate time series [B, C, T]
             (C channels = graph nodes, T frames = time axis)
  * output : sequence embedding [B, out_dim]   (the classifier `linear` is removed,
             the identity is learned with metric losses instead)
  * torch_dct is replaced by an exact orthonormal DCT-II matrix, so no extra dependency.
  * variable-length sequences (test phase, `seqL`) are processed one by one.
  * the dynamic graphs are learnable (straight-through top-k, see LearnableGraph) and the
    unused GIN `mlp` is dropped, so DTFPNet is trained end-to-end with the rest of FreQait.

Layer order / initialisation follow the original code on purpose (see DTFPNet README).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.layer import multi_shallow_embedding, DenseGCNConv2d


class LearnableGraph(multi_shallow_embedding):
    """
    multi_shallow_embedding with a straight-through estimator.

    The original builds a binary top-k adjacency, so emb_s / emb_t never receive a gradient
    (the graph stays at its random init).  Here the forward value is the SAME binary top-k graph,
    but the backward pass goes through sigmoid(emb_s @ emb_t), so the graph is trained jointly.
    """

    def forward(self, device):
        hard = super().forward(device)                                   # [G, N, N] binary (no grad)
        soft = torch.sigmoid(torch.matmul(self.emb_s, self.emb_t))      # [G, N, N]
        return hard + soft - soft.detach()


class DynamicGINConv2d(nn.Module):
    """
    DenseGINConv2d of DTFPNet (backbones/layer.py) without its unused `mlp`
    (it is commented out in the original forward), so every parameter receives a gradient.
    """

    def __init__(self, in_channels, out_channels, groups=1, eps=0., train_eps=True):
        super().__init__()
        if train_eps:
            self.eps = nn.Parameter(torch.tensor([eps]))
        else:
            self.register_buffer('eps', torch.tensor([eps]))

    @staticmethod
    def norm(adj):
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

    def forward(self, x, adj, add_loop=True):
        """x: [B, C, N, F], adj: [G, N, N] -> [B, C, N, F]"""
        B, C, N, _ = x.size()
        G = adj.size(0)
        adj = self.norm(adj).to(x.dtype)
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)                    # [B, C, G, N, F//G]
        out = torch.matmul(adj, x)
        out = torch.cat([out[:, :, :1], out[:, :, 1:] + x[:, :, :-1]], 2)  # dynamic: add previous group
        if add_loop:
            out = (1 + self.eps) * x + out
        return out.transpose(2, 3).reshape(B, C, N, -1)


_DCT_CACHE = {}


def dct_matrix(length, device):
    """Orthonormal DCT-II matrix M [L, L]: dct(x) = x @ M.T, idct(X) = X @ M."""
    key = (length, str(device))
    if key not in _DCT_CACHE:
        n = torch.arange(length, dtype=torch.float32, device=device)
        k = n.unsqueeze(1)
        m = torch.cos(math.pi * (2 * n + 1) * k / (2 * length)) * math.sqrt(2.0 / length)
        m[0] = m[0] / math.sqrt(2.0)
        _DCT_CACHE[key] = m
    return _DCT_CACHE[key]


class DCT_GRU(nn.Module):
    """DCT along time -> GRU over the frequency axis -> inverse DCT.  x: [B, N, L]"""

    def __init__(self, dim, GRU_layers):
        super().__init__()
        self.gru_in = nn.GRU(input_size=dim, hidden_size=dim, num_layers=GRU_layers, batch_first=True)

    def forward(self, x_in):
        B, N, L = x_in.shape
        dtype = x_in.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x = x_in.float()
            M = dct_matrix(L, x.device)
            dct_data = torch.matmul(x, M.t())                    # [B, N, L] (frequency)
            dct_data, _ = self.gru_in(dct_data.transpose(1, 2))  # GRU over frequency components
            x = torch.matmul(dct_data.transpose(1, 2), M)        # inverse DCT
        return x.to(dtype).view(B, N, L)


class DTFPNet(nn.Module):
    """
    Parallel time-domain / frequency-domain dynamic graph network.

    Args:
        num_nodes:     number of channels C of the input series (graph nodes)
        in_dim / hidden_dim / out_dim: widths of the 3 GNN stages
        kern_size:     temporal conv kernel per stage
        kern_size_mid: kernel of the time/frequency merge conv per stage
        groups:        number of time groups (= number of graphs)
        GRU_layers:    layers of the DCT-GRU
        dropout_size:  dropout after each stage
        arch:          'dyGIN2d' | 'dyGCN2d'
    """

    def __init__(self, num_nodes=128, in_dim=32, hidden_dim=64, out_dim=128,
                 kern_size=(7, 5, 3), kern_size_mid=(6, 5, 3), groups=2, GRU_layers=1,
                 dropout_size=(0.4, 0.5, 0.5), arch='dyGIN2d', num_parts=None):
        super().__init__()
        assert len(kern_size) == 3 and len(kern_size_mid) == 3 and len(dropout_size) == 3
        self.num_nodes = num_nodes
        self.num_graphs = groups
        self.out_dim = out_dim
        # part output: the graph nodes (= input channels) are grouped by part, in order
        # (PartwiseProjection concatenates parts), so node groups give part embeddings
        if num_parts:
            assert num_nodes % num_parts == 0, 'num_nodes must be divisible by num_parts'
        self.num_parts = num_parts

        k_neighs = num_nodes
        self.g_constr = LearnableGraph(num_nodes, k_neighs, self.num_graphs)
        self.g_constr_fre = LearnableGraph(num_nodes, k_neighs, self.num_graphs)

        self.asb = DCT_GRU(dim=num_nodes, GRU_layers=GRU_layers)

        gnn_model = {'dyGIN2d': DynamicGINConv2d, 'dyGCN2d': DenseGCNConv2d}[arch]

        self.tconvs_1 = nn.Conv2d(1, in_dim, (1, kern_size[0]), padding="same")
        self.gconvs_1 = gnn_model(in_dim, in_dim, groups)
        self.bns_1 = nn.BatchNorm2d(in_dim)

        self.tconvs_tmp_1 = nn.Conv2d(in_dim * 2, in_dim, (1, kern_size_mid[0]), padding="same")
        self.tconvs_tmp_2 = nn.Conv2d(hidden_dim * 2, hidden_dim, (1, kern_size_mid[1]), padding="same")

        self.tconvs_2 = nn.Conv2d(in_dim, hidden_dim, (1, kern_size[1]), padding="same")
        self.gconvs_2 = gnn_model(hidden_dim, hidden_dim, groups)
        self.bns_2 = nn.BatchNorm2d(hidden_dim)

        self.tconvs_3 = nn.Conv2d(hidden_dim, out_dim, (1, kern_size[2]), padding="same")
        self.gconvs_3 = gnn_model(out_dim, out_dim, groups)
        self.bns_3 = nn.BatchNorm2d(out_dim)

        self.dropout_1, self.dropout_2, self.dropout_3 = dropout_size
        self.activation = nn.ReLU()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def _forward_batch(self, inputs):
        """inputs: [B, N, L] -> [B, out_dim]"""
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            inputs = F.pad(inputs, (int(pad_size), math.ceil(pad_size)), mode='constant', value=0.0)
        fre_asb = self.asb(inputs)
        B, N, L = inputs.size(0), inputs.size(-2), inputs.size(-1)
        inputs = inputs.reshape(B, -1, N, L)
        fre_asb = fre_asb.reshape(B, -1, N, L)
        adj = self.g_constr(inputs.device)
        adj_fre = self.g_constr_fre(inputs.device)

        # stage 1: time branch / frequency branch (shared convs), then merge
        x_1 = self.tconvs_1(inputs)
        x_1 = self.bns_1(self.activation(x_1))
        x_1 = self.gconvs_1(x_1, adj)

        x_fre_1 = self.tconvs_1(fre_asb)
        x_fre_1 = self.gconvs_1(x_fre_1, adj_fre)

        x_1_out = self.tconvs_tmp_1(torch.cat([x_1, x_fre_1], dim=1))
        x_1_out = F.dropout(x_1_out, p=self.dropout_1, training=self.training)

        # stage 2
        x_2 = self.tconvs_2(x_1_out)
        x_2 = self.bns_2(self.activation(x_2))
        x_2 = self.gconvs_2(x_2, adj)

        x_b, x_e, x_n, x_l = x_fre_1.shape
        x_fre = self.asb(x_fre_1.reshape(x_b * x_e, x_n, x_l)).reshape(x_b, x_e, x_n, x_l)
        x_fre_2 = self.tconvs_2(x_fre)
        x_fre_2 = self.gconvs_2(x_fre_2, adj_fre)

        x_2 = self.tconvs_tmp_2(torch.cat([x_2, x_fre_2], dim=1))
        x_2_out = F.dropout(x_2, p=self.dropout_2, training=self.training)

        # stage 3
        x_3 = self.tconvs_3(x_2_out)
        x_3 = self.bns_3(self.activation(x_3))
        x_3 = self.gconvs_3(x_3, adj)
        x_out = F.dropout(x_3, p=self.dropout_3, training=self.training)

        if self.num_parts:
            x_out = x_out.mean(-1)                                           # time  -> [B, out_dim, N]
            return x_out.view(B, self.out_dim, self.num_parts, -1).mean(-1)  # nodes -> [B, out_dim, P]
        return self.global_pool(x_out).view(B, -1, 1)                        # [B, out_dim, 1]

    def forward(self, x, seqL=None):
        """
        x:    [n, C, s]  frame features along time
        seqL: None (fixed-length training batch) or [1, k] lengths of k sequences
              concatenated along s (OpenGait test-time collate).
        out:  [n or k, out_dim, bins]  (bins = 1, or num_parts)
        """
        if seqL is None:
            return self._forward_batch(x)
        lengths = seqL[0].data.cpu().numpy().tolist()
        outs, start = [], 0
        for L in lengths:
            outs.append(self._forward_batch(x[:, :, start:start + L]))
            start += L
        return torch.cat(outs, 0)
