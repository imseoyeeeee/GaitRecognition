import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks
from ..freqait_modules import EFDEncoder, SkeletonParser, PartGCNEncoder, AttentionFusion, PartTransformer
from ..dtfpnet import DTFPNet


class FreQait(BaseModel):
    """
    FreQait: EFD (part-contour frequency) + skeleton gait recognition.

    Inputs (one sequence folder = two .pkl files, sorted order):
        ipts[0]  EFD        [n, s, P=8, H, Co=4]
        ipts[1]  keypoints  [n, s, 17, 3]  (x, y, conf, COCO-17)

    Step 1 (spatial, per frame)
        EFD  -> 1D CNN -> harmonic attention                   [n, s, 8, D]
        pose -> normalize -> parse [8, 3, 3] -> GCN -> joint attention pooling   [n, s, 8, D]
    Step 2 (fusion)
        SkeletonGait++ attention fusion                       [n, s, 8, D]
        transformer over parts (CLS)                          [n, s, C]
    Step 2 (temporal)
        DTFPNet                                               [n, C']
    Head
        FC -> embedding [n, E, 1]  (triplet loss);  optional BNNeck + CE
    """

    def build_network(self, model_cfg):
        P = model_cfg.get('num_parts', 8)
        D = model_cfg.get('dim', 128)
        efd_cfg = model_cfg.get('efd', {})
        skl_cfg = model_cfg.get('skeleton', {})
        fus_cfg = model_cfg.get('fusion', {})
        trf_cfg = model_cfg.get('transformer', {})
        tmp_cfg = dict(model_cfg.get('dtfpnet', {}))
        head_cfg = model_cfg.get('head', {})

        # Step 1 - spatial
        self.efd_encoder = EFDEncoder(num_parts=P, dim=D, **efd_cfg)
        self.skeleton_parser = SkeletonParser(
            parts=skl_cfg.get('parts'), conf_thresh=skl_cfg.get('conf_thresh', 0.1),
            mode=skl_cfg.get('mode', 'data'), edge_sigma=skl_cfg.get('edge_sigma', 8.0),
            canvas_height=skl_cfg.get('canvas_height', 128.0))
        self.skeleton_encoder = PartGCNEncoder(
            num_parts=P, num_joints=self.skeleton_parser.K, hidden_dim=skl_cfg.get('hidden_dim', 64),
            dim=D, dropout=skl_cfg.get('dropout', 0.1))

        # Step 2 - fusion
        self.fusion = AttentionFusion(D, **fus_cfg)
        frame_dim = trf_cfg.get('out_dim', D)
        self.transformer = PartTransformer(num_parts=P, dim=D, **trf_cfg)

        # Step 2 - temporal
        tmp_cfg['num_nodes'] = frame_dim
        self.temporal = DTFPNet(**tmp_cfg)

        # Head
        embed_dim = head_cfg.get('embed_dim', 256)
        self.FCs = SeparateFCs(1, self.temporal.out_dim, embed_dim)
        self.use_bnneck = 'SeparateBNNecks' in model_cfg
        if self.use_bnneck:
            self.BNNecks = SeparateBNNecks(1, embed_dim, class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.inference_use_emb2 = model_cfg.get('use_emb2', False) and self.use_bnneck

    def init_parameters(self):
        # keep PyTorch default initialisation (DTFPNet relies on it, see its README)
        pass

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        efd, pose = ipts[0], ipts[1]
        n, s = efd.shape[:2]
        assert pose.shape[:2] == (n, s), 'EFD and keypoint sequences must have the same length'

        # ---------- Step 1: spatial (frame level) ----------
        efd_feat = self.efd_encoder(efd.reshape(n * s, *efd.shape[2:]))           # [ns, P, D]
        parts, graph, conf = self.skeleton_parser(pose.reshape(n * s, *pose.shape[2:]))
        skl_feat = self.skeleton_encoder(parts, graph, conf)                      # [ns, P, D]

        # ---------- Step 2: fusion + part transformer ----------
        fused = self.fusion(efd_feat, skl_feat)                                   # [ns, P, D]
        frame_feat = self.transformer(fused)                                      # [ns, C]
        frame_feat = frame_feat.view(n, s, -1).transpose(1, 2).contiguous()       # [n, C, s]

        # ---------- Step 2: temporal ----------
        seq_feat = self.temporal(frame_feat, seqL)                                # [n', C']

        # ---------- head ----------
        embed_1 = self.FCs(seq_feat.unsqueeze(-1))                                # [n', E, 1]
        training_feat = {'triplet': {'embeddings': embed_1, 'labels': labs}}
        embed = embed_1
        if self.use_bnneck:
            embed_2, logits = self.BNNecks(embed_1)
            training_feat['softmax'] = {'logits': logits, 'labels': labs}
            if self.inference_use_emb2:
                embed = embed_2

        return {
            'training_feat': training_feat,
            'visual_summary': {},
            'inference_feat': {'embeddings': embed},
        }
