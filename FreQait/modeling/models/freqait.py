import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, SeparateBNNecks
from ..freqait_modules import (EFDEncoder, SkeletonParser, PartGCNEncoder, AttentionFusion, joint_features,
                               PartwiseProjection, PartTransformer, TemporalTransformer, SetPooling)
from ..dtfpnet import DTFPNet
from utils import is_dict


class FreQait(BaseModel):
    """
    FreQait: EFD (part-contour frequency) + skeleton gait recognition.

    Inputs (one sequence = two .pkl files):
        ipts[0]  EFD        [n, s, P=8, H, Co=4]
        ipts[1]  keypoints  [n, s, 17, 3]  (x, y, conf, COCO-17)

    Step 1 (spatial, per frame)
        EFD  -> harmonic selection -> 1D CNN -> harmonic attention                 [n, s, 8, D]
        pose -> normalize -> parse 8 parts (V joint nodes) -> (pos [+ velocity]) -> GCN -> joint attention pool [n, s, 8, D]
    Step 2 (fusion)
        [modality dropout] -> SkeletonGait++ attention fusion                      [n, s, 8, D]
        part aggregation -> [n, s, C]            (part_aggregation.type: partwise | transformer)
    Step 2 (temporal) -> [n, C', bins]           (temporal.type: dtfpnet | transformer | set)
    Head                                         (head.type: global (bins=1) | part (bins=8))
        SeparateFCs -> embedding [n, E, bins] -> triplet;  BNNeck -> CE if CrossEntropyLoss is in loss_cfg

    Every option is selected in model_cfg / loss_cfg, see configs/freqait/freqait_Gait3D.yaml.
    """

    def build_network(self, model_cfg):
        P = model_cfg.get('num_parts', 8)
        D = model_cfg.get('dim', 128)
        efd_cfg = model_cfg.get('efd', {})
        skl_cfg = model_cfg.get('skeleton', {})
        fus_cfg = model_cfg.get('fusion', {})
        aug_cfg = model_cfg.get('augmentation', {}) or {}
        agg_cfg, tmp_cfg = self._convert_legacy_cfg(model_cfg)
        head_cfg = model_cfg.get('head', {})

        # ---------------- Step 1 - spatial ----------------
        self.efd_encoder = EFDEncoder(
            num_parts=P, dim=D, harmonic_drop_prob=aug_cfg.get('harmonic_drop_prob', 0.),
            harmonic_drop_scope=aug_cfg.get('harmonic_drop_scope', 'sequence'),
            harmonic_drop_keep_first=aug_cfg.get('harmonic_drop_keep_first', True), **efd_cfg)
        self.skeleton_parser = SkeletonParser(
            parts=skl_cfg.get('parts'), conf_thresh=skl_cfg.get('conf_thresh', 0.1),
            mode=skl_cfg.get('mode', 'data'), edge_sigma=skl_cfg.get('edge_sigma', 8.0),
            canvas_height=skl_cfg.get('canvas_height', 128.0), edge_conf=skl_cfg.get('edge_conf', False),
            edge_alpha=skl_cfg.get('edge_alpha', 1.0), learn_alpha=skl_cfg.get('learn_alpha', False),
            virtual_require_both=skl_cfg.get('virtual_require_both', True),
            frame_drop_prob=aug_cfg.get('frame_drop_prob', 0.), joint_drop_prob=aug_cfg.get('joint_drop_prob', 0.),
            joint_noise_std=aug_cfg.get('joint_noise_std', 0.))
        if self.skeleton_parser.P != P:
            raise ValueError('model_cfg.skeleton.parts has {} parts but num_parts is {}; give one joint group '
                             'per EFD part, in the same order'.format(self.skeleton_parser.P, P))
        self.use_velocity = skl_cfg.get('use_velocity', False)
        self.skeleton_encoder = PartGCNEncoder(
            part_of_node=self.skeleton_parser.part_of_node.tolist(), in_channels=5 if self.use_velocity else 3,
            hidden_dim=skl_cfg.get('hidden_dim', 64), dim=D, dropout=skl_cfg.get('dropout', 0.1))

        # ---------------- Step 2 - fusion ----------------
        self.modality_drop_prob = aug_cfg.get('modality_drop_prob', 0.)
        self.fusion = AttentionFusion(D, **fus_cfg)
        agg_type = agg_cfg.get('type', 'partwise')
        frame_dim = agg_cfg.get('out_dim', D)
        if agg_type == 'partwise':
            self.part_aggregation = PartwiseProjection(num_parts=P, dim=D, out_dim=frame_dim)
        elif agg_type == 'transformer':
            self.part_aggregation = PartTransformer(num_parts=P, dim=D, out_dim=frame_dim,
                                                    **agg_cfg.get('transformer', {}))
        else:
            raise ValueError("part_aggregation.type must be 'partwise' or 'transformer'")

        # ---------------- Head type ----------------
        head_type = head_cfg.get('type', 'global')
        if head_type not in ('global', 'part'):
            raise ValueError("head.type must be 'global' or 'part'")
        if head_type == 'part' and agg_type != 'partwise':
            raise ValueError("head.type 'part' needs part_aggregation.type 'partwise' "
                             "(the part transformer mixes all parts into every channel).")
        part_bins = P if head_type == 'part' else None

        # ---------------- Step 2 - temporal ----------------
        tmp_type = tmp_cfg.get('type', 'dtfpnet')
        if tmp_type == 'dtfpnet':
            dtf_cfg = dict(tmp_cfg.get('dtfpnet', {}))
            if 'clip_len' not in dtf_cfg:   # default: the training clip length (test-time windows)
                dtf_cfg['clip_len'] = self.cfgs['trainer_cfg']['sampler'].get('frames_num_fixed', 30)
            self.temporal = DTFPNet(num_nodes=frame_dim, num_parts=part_bins, **dtf_cfg)
        elif tmp_type == 'transformer':
            ttf_cfg = dict(tmp_cfg.get('transformer', {}))
            if 'clip_len' not in ttf_cfg:   # default: the training clip length
                ttf_cfg['clip_len'] = self.cfgs['trainer_cfg']['sampler'].get('frames_num_fixed', 30)
            self.temporal = TemporalTransformer(num_nodes=frame_dim, num_parts=part_bins, **ttf_cfg)
        elif tmp_type == 'set':
            self.temporal = SetPooling(num_nodes=frame_dim, num_parts=part_bins)
        else:
            raise ValueError("temporal.type must be 'dtfpnet', 'transformer' or 'set'")

        # ---------------- Head ----------------
        bins = part_bins or 1
        embed_dim = head_cfg.get('embed_dim', 256)
        self.FCs = SeparateFCs(bins, self.temporal.out_dim, embed_dim)

        # losses: the training_feat keys must match the log_prefix of loss_cfg
        loss_cfgs = self.cfgs.get('loss_cfg', [])
        loss_cfgs = [loss_cfgs] if is_dict(loss_cfgs) else list(loss_cfgs)
        prefix = {c['type']: c['log_prefix'] for c in loss_cfgs}
        self.triplet_key = prefix.get('TripletLoss', 'triplet')
        self.ce_key = prefix.get('CrossEntropyLoss')
        self.use_bnneck = self.ce_key is not None
        if self.use_bnneck:
            if 'SeparateBNNecks' not in model_cfg:
                raise ValueError('CrossEntropyLoss needs model_cfg.SeparateBNNecks.class_num (number of training IDs)')
            self.BNNecks = SeparateBNNecks(bins, embed_dim, class_num=model_cfg['SeparateBNNecks']['class_num'])
        self.inference_use_emb2 = model_cfg.get('use_emb2', False) and self.use_bnneck

    def _convert_legacy_cfg(self, model_cfg):
        """
        Old configs had top-level `transformer` (part transformer) and `dtfpnet` keys.
        They are mapped to part_aggregation.type=transformer / temporal.type=dtfpnet,
        which is exactly the old architecture.
        """
        agg_cfg = dict(model_cfg.get('part_aggregation', {}))
        tmp_cfg = dict(model_cfg.get('temporal', {}))
        if 'transformer' in model_cfg and 'part_aggregation' not in model_cfg:
            old = dict(model_cfg['transformer'])
            agg_cfg = {'type': 'transformer', 'out_dim': old.pop('out_dim', model_cfg.get('dim', 128)),
                       'transformer': old}
            self.msg_mgr.log_warning('Legacy config: model_cfg.transformer -> part_aggregation (type: transformer)')
        if 'dtfpnet' in model_cfg and 'temporal' not in model_cfg:
            tmp_cfg = {'type': 'dtfpnet', 'dtfpnet': dict(model_cfg['dtfpnet'])}
            self.msg_mgr.log_warning('Legacy config: model_cfg.dtfpnet -> temporal (type: dtfpnet)')
        unknown = {'transformer', 'dtfpnet'} & set(model_cfg)
        if unknown and 'part_aggregation' in model_cfg and 'temporal' in model_cfg:
            self.msg_mgr.log_warning('Ignored legacy keys {} (part_aggregation / temporal are set).'.format(unknown))
        return agg_cfg, tmp_cfg

    def load_state_dict(self, state_dict, strict=True):
        """Accept checkpoints of the old layouts (`transformer.*` part transformer, GIN `mlp` weights,
        fixed-size parts: joint embedding [P, K, D] and stored skeleton graph buffers)."""
        new_sd = type(state_dict)()
        for k, v in state_dict.items():
            if k.startswith('transformer.'):
                k = 'part_aggregation.' + k[len('transformer.'):]
            if k.startswith('temporal.gconvs_') and '.mlp.' in k:
                continue  # unused GIN mlp of the original DTFPNet layer
            if k.startswith('skeleton_parser.') and k.split('.')[-1] in ('part_index', 'topology', 'edge_graph'):
                continue  # graph buffers are rebuilt from the config
            if k == 'skeleton_encoder.joint_embedding' and v.dim() == 3:
                v = v.reshape(-1, v.shape[-1])  # [P, K, D] -> [V, D]
            new_sd[k] = v
        return super().load_state_dict(new_sd, strict=strict)

    def init_parameters(self):
        # keep PyTorch default initialisation (DTFPNet relies on it, see its README)
        pass

    def modality_dropout(self, efd_feat, skl_feat, n, s):
        """training only: per sequence, drop the EFD or the skeleton features (each with prob p/2)."""
        if not self.training or self.modality_drop_prob <= 0:
            return efd_feat, skl_feat
        r = torch.rand(n, device=efd_feat.device)
        keep_efd = (r >= self.modality_drop_prob / 2).to(efd_feat.dtype)
        keep_skl = ((r < self.modality_drop_prob / 2) | (r >= self.modality_drop_prob)).to(skl_feat.dtype)
        efd_feat = (efd_feat.view(n, s, *efd_feat.shape[1:]) * keep_efd.view(n, 1, 1, 1)).view_as(efd_feat)
        skl_feat = (skl_feat.view(n, s, *skl_feat.shape[1:]) * keep_skl.view(n, 1, 1, 1)).view_as(skl_feat)
        return efd_feat, skl_feat

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        efd, pose = ipts[0], ipts[1]
        n, s = efd.shape[:2]
        assert pose.shape[:2] == (n, s), 'EFD and keypoint sequences must have the same length'

        # ---------- Step 1: spatial (frame level) ----------
        efd_feat = self.efd_encoder(efd.reshape(n * s, *efd.shape[2:]), n)        # [ns, P, D]
        p_xy, p_conf, graph = self.skeleton_parser(pose.reshape(n * s, *pose.shape[2:]))
        feats = joint_features(p_xy, p_conf, n, s, seqL, self.use_velocity)      # [ns, V, 3 or 5]
        skl_feat = self.skeleton_encoder(feats, graph, p_conf)                    # [ns, P, D]

        # ---------- Step 2: fusion + part aggregation ----------
        efd_feat, skl_feat = self.modality_dropout(efd_feat, skl_feat, n, s)
        fused = self.fusion(efd_feat, skl_feat)                                   # [ns, P, D]
        frame_feat = self.part_aggregation(fused)                                 # [ns, C]
        frame_feat = frame_feat.view(n, s, -1).transpose(1, 2).contiguous()       # [n, C, s]

        # ---------- Step 2: temporal ----------
        seq_feat = self.temporal(frame_feat, seqL)                                # [n', C', bins]

        # ---------- head ----------
        embed_1 = self.FCs(seq_feat)                                              # [n', E, bins]
        training_feat = {self.triplet_key: {'embeddings': embed_1, 'labels': labs}}
        embed = embed_1
        if self.use_bnneck:
            embed_2, logits = self.BNNecks(embed_1)
            training_feat[self.ce_key] = {'logits': logits, 'labels': labs}
            if self.inference_use_emb2:
                embed = embed_2

        visual_summary = {}
        if self.training and self.skeleton_parser.learn_alpha:      # tensorboard: scalar/skeleton_alpha/part<i>
            for i, a in enumerate(self.skeleton_parser.alpha().detach()):
                visual_summary['scalar/skeleton_alpha/part{}'.format(i)] = a
        return {
            'training_feat': training_feat,
            'visual_summary': visual_summary,
            'inference_feat': {'embeddings': embed},
        }
