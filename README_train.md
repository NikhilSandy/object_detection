# DINOv3 (Frozen) + DETR Training

This workspace includes `train_dinov3_detr.py` implementing the `Plan.md` pipeline:
- Frozen `facebook/dinov3-vitl16-pretrain-lvd1689m` backbone
- DETR detection transformer head
- YOLO dataset ingestion (`train/images|labels`, `val/images|labels`)
- COCO-style eval metrics each epoch
- Checkpoint saved each epoch
- Optional Trackio logging

## Pilot Run (10 train + 10 val, 10 epochs)

```bash
conda run -n hf_env python train_dinov3_detr.py \
  --phase pilot \
  --num_train_epochs 10 \
  --run_name dinov3-detr-pilot
```

## Full Run (all samples, 10 epochs)

```bash
conda run -n hf_env python train_dinov3_detr.py \
  --phase full \
  --num_train_epochs 10 \
  --run_name dinov3-detr-full
```

## Useful Flags

- `--disable_trackio` to run without Trackio.
- `--encoder_layers 0` to test decoder-only ablation.
- `--shortest_edge 800 --longest_edge 1333 --pad_height 800 --pad_width 1344` for lower memory mode.
- `--max_train_samples N --max_val_samples N` to custom-limit dataset size.

## Trackio CLI

```bash
conda run -n hf_env trackio list projects --json
conda run -n hf_env trackio list runs --project dinov3-detr-crowdhuman --json
```
