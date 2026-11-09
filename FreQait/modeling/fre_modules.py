import torch
import torch.nn as nn
import torch.nn.functional as F

'''
from math import ceil
import Tensor
from torch_dct import dct, idct
from .backbones.layer import DenseGINConv2d, DenseGCNConv2d
'''

def conv1D(in_planes, out_planes, kernel_size=3, padding=1, stride=1):
    """1D convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def build_coco_adjacency():
    num_joints = 17

    edges = [
        (0, 1), (0, 2),
        (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]

    A = torch.eye(num_joints)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # Symmetric degree normalization
    degree = A.sum(dim=1)
    D_inv_sqrt = torch.diag(
        degree.pow(-0.5)
    )

    A = D_inv_sqrt @ A @ D_inv_sqrt

    return A

class HarmonicSelfAttention(nn.Module):
    """
    Input:  [N, H, D]
    Output: [N, H, D]
    """
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x: [N, 32, 128]

        residual = x
        x_norm = self.norm(x)

        attn_out, _ = self.attention(
            x_norm,
            x_norm,
            x_norm
        )

        x = residual + self.dropout(attn_out)

        x = x + self.dropout(self.ffn(x))

        return x


class HarmonicAttentionPooling(nn.Module):
    """
    Input:  [N, H, D]
    Output: [N, D]
    """
    def __init__(self, dim=128):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x: [N, 32, 128]

        weights = self.score(x)
        weights = torch.softmax(weights, dim=1)

        # Weighted sum over harmonic dimension
        out = torch.sum(x * weights, dim=1)

        return out

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, A):
        # x: [N, 17, C]

        x = torch.matmul(A, x)
        x = self.fc(x)
        N, J, C = x.shape
        x = self.bn(
            x.reshape(N * J, C)
        ).reshape(N, J, C)

        x = F.relu(x)
        x = self.dropout(x)
        return x

class GCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn1 = GraphConv(in_channels, out_channels)
        self.gcn2 = GraphConv(out_channels, out_channels)

        if in_channels != out_channels:
            self.residual = nn.Linear(in_channels, out_channels)
        else:
            self.residual = nn.Identity()

    def forward(self, x, A):
        identity = self.residual(x)
        x = self.gcn1(x, A)
        x = self.gcn2(x, A)
        return F.relu(x + identity)

class PartAttentionPooling(nn.Module):

    def __init__(self, dim=128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )
    def forward(self, x, confidence):
        """
        x:
            [N, num_part_joints, 128]
        confidence:
            [N, num_part_joints]
        output:
            [N, 128]
        """

        scores = self.attention(x).squeeze(-1)
        scores = scores + torch.log(confidence.clamp(min=1e-6))
        weights = torch.softmax(scores, dim=1)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return out



class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_parts=8,
        spatial_heads=4,
        temporal_heads=4,
        spatial_layers=2,
        temporal_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.part_embedding = nn.Parameter(
            torch.randn(1, 1, num_parts, embed_dim) * 0.02
        )

        # Spatial Transformer
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=spatial_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer,
            num_layers=spatial_layers
        )

        # Temporal Transformer
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=temporal_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers
        )

        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,T,8,128]

        B, T, P, D = x.shape

        # Part positional embedding
        x = x + self.part_embedding

        # --------------------------------
        # 1. Spatial Transformer
        # --------------------------------

        # [B*T, 8, 128]
        spatial_x = x.reshape(B * T, P, D)

        spatial_x = self.spatial_transformer(spatial_x)

        spatial_x = self.spatial_norm(spatial_x)

        # [B,T,8,128]
        x = spatial_x.reshape(B, T, P, D)

        # --------------------------------
        # 2. Temporal Transformer
        # --------------------------------

        # 각 part별 시간적 관계 학습
        # [B,8,T,128]
        temporal_x = x.permute(0, 2, 1, 3).contiguous()

        # [B*8,T,128]
        temporal_x = temporal_x.reshape(B * P, T, D)

        temporal_x = self.temporal_transformer(temporal_x)

        temporal_x = self.temporal_norm(temporal_x)

        # [B,8,T,128]
        temporal_x = temporal_x.reshape(B, P, T, D)

        # [B,T,8,128]
        x = temporal_x.permute(0, 2, 1, 3).contiguous()

        return x

class GaitEmbedding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()

        # Temporal attention
        self.temporal_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Part attention
        self.part_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B,T,8,128]

        B, T, P, D = x.shape

        # --------------------------------
        # 1. Temporal Attention Pooling
        # --------------------------------

        # [B,8,T,128]
        x = x.permute(0, 2, 1, 3)

        # [B,8,T,1]
        temporal_weights = self.temporal_attention(x)

        temporal_weights = torch.softmax(
            temporal_weights,
            dim=2
        )

        # [B,8,128]
        x = torch.sum(
            x * temporal_weights,
            dim=2
        )

        # --------------------------------
        # 2. Part Attention Pooling
        # --------------------------------

        # [B,8,1]
        part_weights = self.part_attention(x)

        part_weights = torch.softmax(
            part_weights,
            dim=1
        )

        # [B,128]
        x = torch.sum(
            x * part_weights,
            dim=1
        )

        x = self.norm(x)

        return x

'''
class DCT_GRU(nn.Module):
    def __init__(self, dim, GRU_layers):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, dtype=torch.float32))
        self.gru_in = nn.GRU(input_size=dim, hidden_size=dim, num_layers=GRU_layers, batch_first=True)

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply DCT along the time dimension
        dct_data = dct(x)

        dct_data = dct_data.transpose(1, 2)
        dct_data,hi_ = self.gru_in(dct_data)
        dct_data = dct_data.transpose(1, 2)

        # Apply Inverse DCT
        x = idct(dct_data)
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x

class GNNStack(nn.Module):
    """ The stack layers of GNN.
    """
    def __init__(self):
        super().__init__()
        # TODO: Sparsity Analysis
        gnn_model_type = args.arch
        num_layers = args.num_layers
        groups = args.groups
        kern_size = args.kern_size
        kern_size_mid = args.kern_size_mid
        in_dim = args.in_dim
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim

        num_nodes = args.num_channels
        num_classes = args.num_classes

        self.pos_drop = nn.Dropout(p=0.15)
        self.num_graphs = 2
        self.groups=groups
        self.seq_mid_len = 0
        self.num_feats = None #seq_len
        """
        if seq_len % groups:
            self.num_feats += (groups - seq_len % groups)
        self.g_constr = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)  # return adj
        self.g_constr_fre = multi_shallow_embedding(num_nodes, k_neighs, self.num_graphs)  # return adj
        if seq_len % self.num_graphs:
            pad_size = (self.num_graphs - seq_len % self.num_graphs) / 2
            temp_length = F.pad(torch.randn(3,3,4,2,seq_len), (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
            self.seq_len_new = temp_length.size(-1)
        else:
            self.seq_len_new = seq_len
        """
        self.seq_len_new = None
        GRU_layers = 4

        self.asb = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        self.asb_1 = DCT_GRU(dim=num_nodes,GRU_layers=GRU_layers)
        # self.asb = DCT(len=self.seq_len_new)
        # self.asb_1 = DCT(len=self.seq_len_new)
        # self.asb = DFT(dim=self.seq_len_new)
        # self.asb_1 = DFT(dim=self.seq_len_new)

        gnn_model, heads = self.build_gnn_model(gnn_model_type)  # gnn_model=DenseGINConv2d  heads=1

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        # To make the framework presented in this paper more accessible, the code below are deliberately kept concise and avoid loop constructs. 
        # Furthermore, as superior performance compared to SoTA methods was already achieved using only three layers, 
        # we did not investigate whether stacking deeper architectures could yield further performance improvements.
        self.tconvs_1 = nn.Conv2d(1, in_dim, (1, kern_size[0]), padding="same")
        self.gconvs_1 = gnn_model(in_dim, heads * in_dim, groups)
        self.bns_1 = nn.BatchNorm2d(heads * in_dim)

        self.tconvs_tmp_1 = nn.Conv2d(in_dim*2, in_dim, (1, kern_size_mid[0]), padding="same")
        self.tconvs_tmp_2 = nn.Conv2d(hidden_dim*2, hidden_dim, (1, kern_size_mid[1]), padding="same")
        self.tconvs_tmp_3 = nn.Conv2d(out_dim*2, out_dim, (1, kern_size_mid[2]), padding="same")

        self.tconvs_2 = nn.Conv2d(in_dim, hidden_dim, (1, kern_size[1]), padding="same")
        self.gconvs_2 = gnn_model(hidden_dim, heads * hidden_dim, groups)
        self.bns_2 = nn.BatchNorm2d(heads * hidden_dim)

        self.tconvs_3 = nn.Conv2d(hidden_dim, out_dim, (1, kern_size[2]), padding="same")
        self.gconvs_3 = gnn_model(out_dim, heads * out_dim, groups)
        self.bns_3 = nn.BatchNorm2d(heads * out_dim)

        self.linear = nn.Linear(heads * out_dim, num_classes)

        self.dropout_1 = 0.5
        self.dropout_2 = 0.5
        self.dropout_3 = 0.5
        self.activation = nn.ReLU()  # LeakyReLU()  

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.reset_parameters()

    def reset_parameters(self):
        self.tconvs_1.reset_parameters()
        self.gconvs_1.reset_parameters()
        self.bns_1.reset_parameters()
        self.tconvs_2.reset_parameters()
        self.gconvs_2.reset_parameters()
        self.bns_2.reset_parameters()
        self.tconvs_3.reset_parameters()
        self.gconvs_3.reset_parameters()
        self.bns_3.reset_parameters()
        self.tconvs_tmp_1.reset_parameters()
        self.tconvs_tmp_2.reset_parameters()
        self.tconvs_tmp_3.reset_parameters()
        self.linear.reset_parameters()
        self.g_constr.reset_parameters()
        self.g_constr_fre.reset_parameters()

    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1

    def pretrain(self, x_in):
        return x_in

    def forward(self, inputs: Tensor):
        # inputs = self.pos_drop(inputs)  # Heartbeat  ArrowHead ...
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            inputs = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            inputs = inputs
        fre_asb = self.asb(inputs)
        B = inputs.size(0)
        _ = inputs.size(1)
        N = inputs.size(-2)
        L = inputs.size(-1)
        inputs = inputs.reshape(B, -1, N, L)
        fre_asb = fre_asb.reshape(B, -1, N, L)
        adj = self.g_constr(inputs.device)
        adj_fre = self.g_constr_fre(inputs.device)

        x_1 = self.tconvs_1(inputs)
        x_1 = self.bns_1(self.activation(x_1))
        x_1 = self.gconvs_1(x_1,adj)

        x_fre_1 = self.tconvs_1(fre_asb)
        x_fre_1 = self.gconvs_1(x_fre_1,adj_fre)

        x_1_out = torch.cat([x_1,x_fre_1],dim=1)
        x_1_out = self.tconvs_tmp_1(x_1_out)

        x_1_out = F.dropout(x_1_out, p=self.dropout_1, training=self.training)

        x_2 = self.tconvs_2(x_1_out)
        x_2 = self.bns_2(self.activation(x_2))
        x_2 = self.gconvs_2(x_2,adj)

        x_b,x_e,x_n,x_l = x_fre_1.size(0),x_fre_1.size(1),x_fre_1.size(2),x_fre_1.size(-1)
        fre_asb = x_fre_1.reshape(x_b*x_e, x_n, x_l)
        fre_asb = self.asb(fre_asb)
        x_fre = fre_asb.reshape(x_b,x_e,x_n,x_l)

        x_fre_2 = self.tconvs_2(x_fre)
        x_fre_2 = self.gconvs_2(x_fre_2,adj_fre)

        x_2 = torch.cat([x_2,x_fre_2],dim=1)
        x_2 = self.tconvs_tmp_2(x_2)
        x_2_out = F.dropout(x_2, p=self.dropout_2, training=self.training)

        x_3 = self.tconvs_3(x_2_out)
        x_3 = self.bns_3(self.activation(x_3))
        x_3 = self.gconvs_3(x_3,adj)

        # x_b,x_e,x_n,x_l = x_fre_2.size(0),x_fre_2.size(1),x_fre_2.size(2),x_fre_2.size(-1)
        # fre_asb = x_fre_2.reshape(x_b*x_e, x_n, x_l)
        # fre_asb = self.asb_1(fre_asb)
        # x_fre = fre_asb.reshape(x_b,x_e,x_n,x_l)
        #
        # x_fre = self.tconvs_3(x_fre)
        # x_fre = self.gconvs_3(x_fre,adj_fre)
        #
        # x_3 = torch.cat([x_3,x_fre],dim=1)
        # x_3 = self.tconvs_tmp_3(x_3)

        x_out = F.dropout(x_3, p=self.dropout_3, training=self.training)

        out = self.global_pool(x_out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
'''