# DINOv3 (Frozen) + DETR Training Plan

## 0. Objective
Train a DETR-style object detector for `person` and `head` on:
- `/home/awiros-tech/Projects/datasets/crowd_human_mot_dataset`

Using:
- Frozen DINOv3 backbone
- Trainable DETR decoder/head
- Trackio for training visibility
- Checkpoint every epoch
- Eval every epoch with standard object-detection metrics

---

## 1. Verified Context

### 1.1 Dataset (reconfirmed)
- Format: YOLO (`class cx cy w h`, normalized)
- Structure:
  - `train/images`, `train/labels`
  - `val/images`, `val/labels`
- Classes from `data.yaml`:
  - `0: person`
  - `1: head`
- Scale:
  - Train images: `55,442`
  - Val images: `12,065`
  - Train boxes: `4,738,409` (~85.47/image)
  - Val boxes: `961,208` (~79.67/image)

### 1.2 System Specs
- GPU: `NVIDIA RTX 4000 SFF Ada`, `20 GB VRAM`
- RAM: `30 GB`
- CPU: `Intel i7-13700`, `24 threads`
- Free disk: ~`195 GB`
- CUDA available, BF16 supported

### 1.3 Model choice for this plan
- Backbone target: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Rationale:
  - Strong dense features for small crowded objects
  - Better quality/compute tradeoff than H+/7B
  - Fits local hardware with frozen-backbone DETR setup

---

## 2. Hugging Face Skills + MCP Usage Plan

### 2.1 `hf-mcp` skill
Use for:
- Verifying latest Transformers docs/APIs
- Confirming supported DETR + DINOv3 backbone replacement path
- Optional: model/dataset card checks during iteration

Primary MCP tools:
- `hf_doc_search`, `hf_doc_fetch`
- `hub_repo_details`
- `hf_whoami`

### 2.2 `hugging-face-trackio` skill
Use for:
- Logging and visualizing run progress
- Local or HF Space-backed dashboard monitoring
- Run/metric retrieval via CLI

### 2.3 `hugging-face-jobs` skill (optional fallback)
Use if local runtime becomes too long/unstable:
- Launch same training workflow on HF Jobs GPU
- Persist checkpoints/artifacts to Hub
- Keep Trackio reporting active via `space_id`

---

## 3. Environment Setup

Environment:
- Conda env: `hf_env`

Install/ensure packages:
- `transformers`, `torch`, `torchvision`, `timm`, `accelerate`
- `scipy` (required by Hungarian matcher)
- `pycocotools`
- `torchmetrics`
- `albumentations`
- `datasets`
- `evaluate`
- `trackio`
- `huggingface_hub`

Validation checks:
1. Import all libs successfully.
2. Confirm DINOv3 gated access works.
3. Confirm `DetrForObjectDetection` + DINOv3 backbone replacement runs.

---

## 4. Data Pipeline Plan

### 4.1 Annotation conversion strategy
- Keep source dataset in YOLO format.
- Build an on-the-fly adapter to COCO-style annotations expected by DETR image processor/trainer.
- Preserve both classes (`person`, `head`).
- Handle images without label file as empty annotation lists.

### 4.2 Box conversions
- Read YOLO labels as normalized `(cx, cy, w, h)`.
- Convert to absolute coordinates per image size.
- Build COCO-style `annotations` with `bbox=[x, y, w, h]`, `area`, `category_id`, `iscrowd=0`.
- Preserve empty-image samples by using `annotations=[]`.

### 4.3 Preprocessing and augmentation (explicit)
- Base image processor settings:
  - `size={"shortest_edge": 960, "longest_edge": 1536}`
  - `do_pad=True`
  - `pad_size={"height": 960, "width": 1536}`
- Train-time augmentation (mild, small-object-safe):
  - `HorizontalFlip(p=0.5)`
  - `Affine(rotate=(-5, 5), shear=(-3, 3), translate_percent=(0.03, 0.03), scale=(0.9, 1.1), p=0.4)`
  - Optional low-strength brightness/contrast jitter.
- Eval-time augmentation:
  - No random transforms (resize + normalize + pad only).
- Avoid random crops and heavy geometric distortion to protect tiny head boxes.

---

## 5. Model Architecture Plan

### 5.1 Base model
- `DetrForObjectDetection`
- Backbone replaced with:
  - `AutoBackbone.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")`

### 5.2 Freeze policy
- Freeze full backbone:
  - `model.model.freeze_backbone()`
- Train only DETR encoder/decoder heads and prediction heads.

### 5.3 Query count
- Set `num_queries=300` for crowded scenes.

### 5.4 Encoder/decoder configuration
- Default (recommended): keep DETR transformer encoder + decoder intact.
  - `encoder_layers=6` (default)
- Optional variant: remove DETR encoder and keep decoder-only adaptation on top of DINOv3 features.
  - `encoder_layers=0`
  - Keep this as an ablation, not default.

### 5.5 Appendix D.9 `should include` pointers
- Keep a frozen DINOv3 backbone and train a separate DETR transformer detection stack.
  - Pointer: DINOv3 paper, Appendix D.9 (object-detection transfer setup).
- Keep DETR encoder + decoder enabled in the default pipeline (`encoder_layers=6` with decoder active).
  - Pointer: DINOv3 paper, Appendix D.9 (detection transformer configuration).
- Keep standard DETR matching/loss behavior (Hungarian matching + class/box objectives).
  - Pointer: DINOv3 paper, Appendix D.9 (DETR-style detector training recipe).

---

## 6. Training Configuration (Hardware-Aware)

Initial local config (safe starting point):
- Precision: `bf16=True` (fallback `fp16=True` if needed)
- `per_device_train_batch_size=2`
- `per_device_eval_batch_size=2`
- `gradient_accumulation_steps=4` (effective batch size 8)
- `dataloader_num_workers=8` (tune 6-10)
- Learning rate (trainable decoder/head): `1e-4`
- Weight decay: `1e-4`
- Warmup ratio: `0.05`
- Gradient clip: `0.1`
- `remove_unused_columns=False`

Image size policy:
- Default: shortest edge `960`, longest edge `1536`, pad to `960x1536`.
- OOM fallback: shortest edge `800`, longest edge `1333`, pad to nearest practical width (`1344`).

Processor baseline:
- `DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", size={"shortest_edge": 960, "longest_edge": 1536}, do_pad=True, pad_size={"height": 960, "width": 1536})`

---

## 7. Checkpointing + Evaluation Policy (Required)

Configure Trainer:
- `save_strategy="epoch"` (checkpoint after every epoch)
- `evaluation_strategy="epoch"` (evaluate after every epoch)
- `load_best_model_at_end=True`
- `metric_for_best_model="eval_map"`
- `greater_is_better=True`
- `save_total_limit=8` (adjustable)

---

## 8. Metrics Plan (Standard Object Detection)

Compute and log COCO-style metrics via `torchmetrics.detection.MeanAveragePrecision`:
- `mAP` (IoU 0.50:0.95)
- `mAP50`, `mAP75`
- `mAP_small`, `mAP_medium`, `mAP_large`
- `mAR_1`, `mAR_10`, `mAR_100`
- `mAR_small`, `mAR_medium`, `mAR_large`
- Per-class AP/AR:
  - `person`
  - `head`

---

## 9. Trackio Monitoring Plan (Required)

### 9.1 Run logging
- Initialize:
  - `trackio.init(project=..., name=..., config=..., [space_id=...])`
- Trainer args:
  - `report_to="trackio"`
  - `run_name=...`
- Finalize:
  - `trackio.finish()`

### 9.2 Dashboard options
- Local:
  - `trackio show --project <project>`
- Persistent/shareable HF Space:
  - Provide `space_id="NikhilSandy/<trackio-space>"`

### 9.3 Metric retrieval
- Use CLI for JSON summaries:
  - `trackio list projects --json`
  - `trackio list runs --project <project> --json`
  - `trackio get run --project <project> --run <run> --json`
  - `trackio get metric --project <project> --run <run> --metric <metric> --json`

---

## 10. Execution Phases

### Phase A: Pilot sanity run
- Use a subset:
  - 10 samples from train split
  - 10 samples from test split (use `val` as test in this dataset)
- 1 epoch only
- Goal:
  - Verify data conversion, metrics, checkpoints, Trackio logging
  - Validate throughput/memory and query count behavior

### Phase B: Full training
- Full train/val splits
- Train for 10 epochs
- Eval each epoch
- Save checkpoint each epoch

---

## 11. Optional Cloud Fallback (HF Jobs)

Trigger conditions:
- Local epoch time is consistently too high.
- Repeated instability due to local resource contention.

Execution:
- Use `hugging-face-jobs` skill.
- Launch with `hf_jobs(...)` on `a10g-large` (or higher if needed).
- Pass `secrets={"HF_TOKEN": "$HF_TOKEN"}`.
- Persist checkpoints/artifacts to Hub every epoch.
- Keep `trackio` active via `space_id` for remote dashboard continuity.

---

## 12. Deliverables

1. Training script: frozen DINOv3 + DETR decoder/head.
2. Data adapter: YOLO -> COCO annotations.
3. Trainer config with:
   - Eval per epoch
   - Save per epoch
   - Best model selection by `eval_map`
4. Trackio-integrated logging and dashboard.
5. Final evaluation report with COCO metrics and per-class scores.
