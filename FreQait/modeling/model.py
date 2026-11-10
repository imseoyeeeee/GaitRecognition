import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .modules import SeparateFCs
from .fre_modules import build_coco_adjacency, conv1D, HarmonicSelfAttention, HarmonicAttentionPooling, GCNBlock, PartAttentionPooling, SpatialTemporalTransformer

from einops import rearrange

class FreQait(BaseModel):

   def build_network(self, model_cfg):
       #B, C = [1, 4, 4, 1], 2
       in_C, B, C = model_cfg['Backbone']['in_channels'], model_cfg['Backbone']['blocks'], model_cfg['Backbone']['C']
       self.inference_use_emb = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False

       self.inplanes = 32 * C

       #EFD Backbone
       self.sil_conv0 = nn.Sequential(conv1D(in_C, 64, 3, 1, 1), nn.BatchNorm1d(self.inplanes), nn.ReLU(inplace=True))
       self.cnn = nn.Sequential(
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.GELU(),

            nn.Conv1d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU())
       self.pos_embedding = nn.Parameter(torch.randn(1, 32, 128) * 0.02)
       self.harmonic_attention = HarmonicSelfAttention(dim=128 , num_heads=4, dropout=0.1)
       self.harmonic_pooling = HarmonicAttentionPooling(dim=128)

       self.fusion = AttentionFusion(32 * C)
       
       self.spatial_temporal_transformer = SpatialTemporalTransformer(dim=128, num_parts=8, num_heads=4, num_layers=2, dropout=0.1)
       
       self.layer2 = nn.Sequential(
           nn.Linear(in_channels=128, out_features=64, bias=True),
           nn.LayerNorm(64),
           nn.ReLU(inplace=True),

           nn.Linear(64, 32),
           nn.LayerNorm(32),
           nn.ReLU(inplace=True))

       self.FCs = SeparateFCs(in_channels=32, out_channels=1, num_parts=8, bias=True)
        


   def inputs_pretreament(self, inputs):
       ### Ensure the same data augmentation for heatmap and silhouette
       pose_sils = inputs[0]
       new_data_list = []
       for pose, sil in zip(pose_sils[0], pose_sils[1]):
           sil = sil[:, np.newaxis, ...] # [T, 1, H, W]
           pose_h, pose_w = pose.shape[-2], pose.shape[-1]
           sil_h, sil_w = sil.shape[-2], sil.shape[-1]
           if sil_h != sil_w and pose_h == pose_w:
               cutting = (sil_h - sil_w) // 2
               pose = pose[..., cutting:-cutting]
           cat_data = np.concatenate([pose, sil], axis=1) # [T, 3, H, W]
           new_data_list.append(cat_data)
       new_inputs = [[new_data_list], inputs[1], inputs[2], inputs[3], inputs[4]]
       return super().inputs_pretreament(new_inputs)

   def forward(self, inputs):
       ipts, labs, _, _, seqL = inputs

       pose = ipts[0]
       pose = pose.transpose(1, 2).contiguous()
       assert pose.size(-1) in [44, 48, 88, 96]
       maps = pose[:, :2, ...]
       sils = pose[:, -1, ...].unsqueeze(1)

       del ipts
       sil = self.sil_conv0(sils)
       sil = self.cnn(sil)
       sil = sil.transpose(1, 2)
       sil = sil + self.pos_embedding
       sil = self.harmonic_attention(sil)
       sil = self.harmonic_pooling(sil)
       sil = sil.reshape(sil.size(0), sil.size(1), 8, 128)

       map = self.SB(maps)
       out1 = self.fusion(sil, map)

       out2 = self.layer2(out1)
       out3 = self.FCs(out2)

       out3 = out3.transpose(1, 2).contiguous()  # [n, p, t]

       # Temporal Pooling, TP
       outs = self.TP(out3, seqL, options={"dim": 2})[0]  # [n, c, h, w]
       n, c, h, w = outs.size()

       # Horizontal Pooling Matching, HPM
       feat = self.HPP(outs)  # [n, c, p]

       embed_1 = self.FCs(feat)  # [n, c, p]
       embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
       
       if self.inference_use_emb:
            embed = embed_2
       else:
            embed = embed_1

       retval = {
           'training_feat': {
               'triplet': {'embeddings': embed_1, 'labels': labs},
               'softmax': {'logits': logits, 'labels': labs}
           },
           'visual_summary': {
               'image/sils': rearrange(pose * 255., 'n c s h w -> (n s) c h w'),
           },
           'inference_feat': {
               'embeddings': embed
           }
       }
       return retval

class AttentionFusion(nn.Module):
    """
    Part-wise EFD-Skeleton Attention Fusion
    EFD input: [B, T, 8, 128]
    Skeleton input: [B, T, 8, 128]
    Output: [B, T, 8, 128]
    """
    def __init__(self, in_channels=128, squeeze_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = max(in_channels // squeeze_ratio, 1)

        # Generate modality attention scores
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_channels * 2)
        )

    def forward(self, efd_feat, skeleton_feat, frame_mask=None, return_attention=False):
        """
        efd_feat: [B, T, 8, 128]
        skeleton_feat: [B, T, 8, 128]
        frame_mask: [B, T]
            True = valid frame
            False = padded frame
        """
        # Check input dimensions
        assert efd_feat.shape == skeleton_feat.shape
        B, T, P, C = efd_feat.shape
        # 1. Concatenate modalities
        # [B, T, 8, 256]
        feats = torch.cat([efd_feat, skeleton_feat], dim=-1)

        # 2. Generate attention scores
        # [B, T, 8, 256]
        scores = self.attention(feats)
        # 3. Separate modality dimension
        # [B, T, 8, 2, 128]
        scores = scores.reshape(B, T, P, 2, C)
        # 4. Softmax over modality dimension
        weights = F.softmax(scores, dim=3)
        # 5. Weighted fusion
        # [B, T, 8, 128]
        fused_feat = (efd_feat * weights[:, :, :, 0, :]
            + skeleton_feat * weights[:, :, :, 1, :])
        # 6. Mask padded frames
        if frame_mask is not None:
            fused_feat = fused_feat * (
                frame_mask[:, :, None, None].to(fused_feat.dtype))

        if return_attention:
            return fused_feat, weights

        return fused_feat

class SkeletonBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, hidden_dim=64, num_joints=17, part_groups=None):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        if part_groups is None:
            part_groups = [
                [1, 10, 13],  # Full Body
                [0, 14, 15],  # Head
                [1, 8, 10],   # Torso
                [5, 6, 7],    # Left arm
                [2, 3, 4],    # Right arm
                [11, 12, 13],    # Left leg
                [8, 9, 10],      # Right leg
                [1, 9, 12] #dress
            ]

        self.part_groups = part_groups
        # Register adjacency as model buffer
        self.register_buffer("A", build_coco_adjacency())
        # Joint feature embedding
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        # Spatial GCN
        self.gcn1 = GCNBlock(hidden_dim, hidden_dim)
        self.gcn2 = GCNBlock(hidden_dim, embed_dim)
        # Shared attention pooling for each part
        self.part_pool = PartAttentionPooling(embed_dim)

    def forward(self, x, frame_mask=None):
        """
        x:
            [B, T, 17, 3]
            (x, y, confidence)
        frame_mask:
            [B, T]
            True = valid frame
            False = padded frame
        output:
            [B, T, 8, 128]
        """

        B, T, J, C = x.shape

        assert J == self.num_joints
        assert C == 3

        # Separate confidence
        coords = x[..., :2]
        confidence = x[..., 2].clamp(0, 1)
        # Confidence-weighted coordinates
        coords = coords * confidence.unsqueeze(-1)
        # [B, T, 17, 3]
        # Keep x, y and confidence as input features
        features = torch.cat([coords, confidence.unsqueeze(-1)], dim=-1)
        # Merge batch and time
        features = features.reshape(B * T, J, 3)
        conf = confidence.reshape(B * T, J)

        # Initial projection
        features = self.input_proj(features)
        # Spatial GCN
        features = self.gcn1(features, self.A)
        features = self.gcn2(features, self.A)

        # Part-wise pooling
        part_features = []
        for group in self.part_groups:
            joint_features = features[:, group, :]
            joint_confidence = conf[:, group]
            pooled = self.part_pool(joint_features, joint_confidence)
            part_features.append(pooled)

        # [B*T, 8, 128]
        out = torch.stack(part_features, dim=1)
        # Restore batch and time
        out = out.reshape(B, T, len(self.part_groups), self.embed_dim)
        # Zero-out padded frames
        if frame_mask is not None:
            out = out * frame_mask[:, :, None, None].to(out.dtype)

        return out