# FreQait

EFD (part-contour frequency features) + skeleton gait recognition, built on the OpenGait training framework.

```
Silhouette EFD [T,8,H,4] ─ 1D CNN ─ Harmonic Attention ───────────────┐
                                                                       ├─ Attention Fusion ─ Transformer ─ DTFPNet ─ FC ─ [B, E]
Keypoints [T,17,3] ─ normalize ─ parse [T,8,3,3] ─ GCN ─ Joint Attn Pool ┘   [B,T,8,128]      [B,T,C]     [B,C]
                     (edge graph [8,3,3], sigma=8.0)
```

| File | Content |
|---|---|
| `modeling/models/freqait.py` | `FreQait` model (BaseModel) |
| `modeling/freqait_modules.py` | EFD encoder, skeleton parser/GCN, attention fusion, part transformer |
| `modeling/dtfpnet.py` | DTFPNet temporal head (port of `DTFPNet-DAAC/DTFPNet_temp`) |
| `evaluation/evaluator.py` | `evaluate_probe_gallery` (Rank-k / mAP / mINP) |
| `configs/freqait/freqait_Gait3D.yaml` | config |
| `datasets/merge_efd_pose.py` | merges EFD + pose pkls into the OpenGait layout |

## 1. Data

Option A: give one folder per modality (order = `[EFD, keypoints]`). Sequences are matched by `<ID>/<CAM>/<SEQ>`:
```yaml
data_cfg:
  dataset_root:
    - <...>/Gait3D/efd    # <ID>/<CAM>/<SEQ>/<SEQ>.pkl  or  <ID>/<CAM>/<SEQ>.pkl
    - <...>/Gait3D/pose
  max_len_diff: 5         # frame numbers are trimmed to the shorter one; a warning is logged above this
```
A sequence must exist in every root (otherwise it is skipped and the count is logged).

Option B: merge both into a single `dataset_root`:
```
python datasets/merge_efd_pose.py \
  --efd_root  <EFD-Analysis output>/EFD8 \
  --pose_root <pose pkl root> \
  --out_root  <dataset_root>
```
Result: `<dataset_root>/<ID>/<CAM>/<SEQ>/00-efd.pkl` (`[T, 8, H, 4]`) and `01-pose.pkl` (`[T, 17, 3]`, COCO-17 x/y/conf).
The partition json needs `TRAIN_SET` / `TEST_SET` (and `PROBE_SET` when using `probe_json`).

## 2. Train / test  (run from this folder)

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
  main.py --cfgs configs/freqait/freqait_Gait3D.yaml --phase train --log_to_file

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
  main.py --cfgs configs/freqait/freqait_Gait3D.yaml --phase test
```
`evaluator_cfg.sampler.batch_size` must equal the number of GPUs.

## 3. Probe-gallery protocol

`evaluator_cfg.probe_gallery` either takes an explicit probe list (`probe_json`, Gait3D style: `"ID-CAM-SEQ"` entries)
or selects probe / gallery sequences by type/view (`probe: {types: [...]}`, `gallery: {types: [...]}`).
Options: `exclude_same_view`, `per_view`, `ranks`.

## Notes

* EFD input: EFD-Analysis now writes non-normalized coefficients by default (`EFD{n}-unnorm/`, limb swing angle kept, start point aligned, divided by body height). Point `dataset_root` at that folder.
* Loss: triplet + BNNeck/CE by default. Removing the `CrossEntropyLoss` entry from `loss_cfg` gives triplet only; the BNNeck is built only when that entry exists (`SeparateBNNecks.class_num` = number of training IDs).
* `efd.harmonics`: an int `n` uses harmonics `0..n-1`; a list uses only those indices (0 = 1st harmonic). Indices refer to the stored harmonic axis (128 with the default EFD-Analysis config).
* `skeleton.use_velocity`: adds joint velocity (frame difference × confidence, zero at sequence starts) to the GCN input.
* `augmentation` (training only, 0 disables): `frame_drop_prob`, `joint_drop_prob`, `joint_noise_std` on the skeleton, and `modality_drop_prob`, which zeroes the EFD or skeleton features of a sequence.
* `head.type`: `global` (one embedding) or `part` (one embedding per part, 8 bins). `part` needs `part_aggregation.type: partwise` and `temporal.type: dtfpnet | set`.
* Inputs are always the EFD tensor and keypoint coordinates (`NoOperation` transforms); no silhouette / heatmap image goes into the network.
* Skeleton ablation, `model_cfg.skeleton.mode`:
  * `data` (default): fixed edge graph `[8, 3, 3]` (chain topology inside each part + self loops).
  * `image`: SkeletonGait heatmap convention. Joint distances are measured with the body scaled to 128 px (`canvas_height`), and a Gaussian `exp(-d²/2σ²)` with σ = 8 (`edge_sigma`) is added to the chain topology, giving a per-frame graph `[N, 8, 3, 3]`.
* Part aggregation ablation, `model_cfg.part_aggregation.type`: `partwise` (default; per-part Linear(128 → C/8) + concat, so DTFPNet's channel graph learns inter-part coordination over time) or `transformer` (transformer over the 8 parts + CLS, per frame).
* `part_aggregation.out_dim` (C, = DTFPNet graph nodes) is 64 by default. DTFPNet's graph convolutions grow ~C², and C=128 is about 2.3× slower.
* Temporal ablation, `model_cfg.temporal.type`: `dtfpnet` (default), `set` (max over frames, GaitSet-style baseline) or `transformer` (temporal transformer with sinusoidal positions + mean pooling). At test time the transformer splits sequences longer than `clip_len` (default = `frames_num_fixed`) into windows (step `clip_stride`) and averages the window embeddings. Set `clip_len: null` to feed the whole sequence.
* Legacy configs (top-level `transformer` / `dtfpnet` keys) are converted to `part_aggregation.type: transformer` + `temporal.type: dtfpnet`, and old checkpoints (`transformer.*`, GIN `mlp`) still load.
* DTFPNet is trained end-to-end. Its top-k graphs use a straight-through estimator: the forward pass is the original binary graph, and gradients flow to the graph embeddings. All parameters are used, so `find_unused_parameters: false`.
