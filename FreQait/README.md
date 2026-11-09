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

* Loss: triplet only by default. For BNNeck + CE (as in SkeletonGait++) uncomment the CE entry in `loss_cfg` and `model_cfg.SeparateBNNecks`.
* Inputs are always the EFD tensor and keypoint coordinates (`NoOperation` transforms); no silhouette / heatmap image goes into the network.
* Skeleton ablation, `model_cfg.skeleton.mode`:
  * `data` (default): fixed edge graph `[8, 3, 3]` (chain topology inside each part + self loops).
  * `image`: SkeletonGait heatmap convention. Joint distances are measured with the body scaled to 128 px (`canvas_height`), and a Gaussian `exp(-d²/2σ²)` with σ = 8 (`edge_sigma`) is added to the chain topology, giving a per-frame graph `[N, 8, 3, 3]`.
* DTFPNet is trained end-to-end. Its top-k graphs use a straight-through estimator: the forward pass is the original binary graph, and gradients flow to the graph embeddings. All parameters are used, so `find_unused_parameters: false`.
