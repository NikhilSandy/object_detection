#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-hf_env}"
DATASET_ROOT="${DATASET_ROOT:-/home/awiros-tech/Projects/datasets/crowd_human_mot_dataset}"
PHASE="${PHASE:-full}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
RUN_NAME="${RUN_NAME:-dinov3-detr-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

BACKBONE_NAME="${BACKBONE_NAME:-facebook/dinov3-vitl16-pretrain-lvd1689m}"
IMAGE_PROCESSOR_NAME="${IMAGE_PROCESSOR_NAME:-facebook/detr-resnet-50}"
DETR_INIT_MODEL="${DETR_INIT_MODEL:-facebook/detr-resnet-50}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-bf16}"

LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.1}"

SHORTEST_EDGE="${SHORTEST_EDGE:-960}"
LONGEST_EDGE="${LONGEST_EDGE:-1536}"
PAD_HEIGHT="${PAD_HEIGHT:-960}"
PAD_WIDTH="${PAD_WIDTH:-1536}"

NUM_QUERIES="${NUM_QUERIES:-300}"
ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
DECODER_LAYERS="${DECODER_LAYERS:-6}"

SEED="${SEED:-42}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

TRACKIO_PROJECT="${TRACKIO_PROJECT:-dinov3-detr-crowdhuman}"
TRACKIO_SPACE_ID="${TRACKIO_SPACE_ID:-}"
DISABLE_TRACKIO="${DISABLE_TRACKIO:-0}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

CMD=(
  conda run --no-capture-output -n "$ENV_NAME" env PYTHONUNBUFFERED=1 python train_dinov3_detr.py
  --dataset_root "$DATASET_ROOT"
  --phase "$PHASE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --run_name "$RUN_NAME"
  --output_dir "$OUTPUT_DIR"
  --backbone_name "$BACKBONE_NAME"
  --image_processor_name "$IMAGE_PROCESSOR_NAME"
  --detr_init_model "$DETR_INIT_MODEL"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --num_workers "$NUM_WORKERS"
  --precision "$PRECISION"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --warmup_ratio "$WARMUP_RATIO"
  --max_grad_norm "$MAX_GRAD_NORM"
  --shortest_edge "$SHORTEST_EDGE"
  --longest_edge "$LONGEST_EDGE"
  --pad_height "$PAD_HEIGHT"
  --pad_width "$PAD_WIDTH"
  --num_queries "$NUM_QUERIES"
  --encoder_layers "$ENCODER_LAYERS"
  --decoder_layers "$DECODER_LAYERS"
  --seed "$SEED"
  --log_every_steps "$LOG_EVERY_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --trackio_project "$TRACKIO_PROJECT"
)

if [[ -n "$TRACKIO_SPACE_ID" ]]; then
  CMD+=(--trackio_space_id "$TRACKIO_SPACE_ID")
fi

if [[ "$DISABLE_TRACKIO" == "1" ]]; then
  CMD+=(--disable_trackio)
fi

if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  CMD+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "$MAX_VAL_SAMPLES" ]]; then
  CMD+=(--max_val_samples "$MAX_VAL_SAMPLES")
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  CMD+=(--local_files_only)
fi

CMD+=("$@")

echo "Launching training with effective train batch size: $((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Command: ${CMD[*]}"
"${CMD[@]}"
