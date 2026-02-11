#!/usr/bin/env python3
"""
Train a frozen DINOv3 + DETR object detector on a YOLO-format dataset.

Implements:
- YOLO -> DETR target conversion
- Frozen DINOv3 backbone + trainable DETR encoder/decoder/head
- Epoch-wise checkpointing
- Epoch-wise evaluation with COCO-style metrics (mAP/mAR)
- Optional Trackio logging
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_convert
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm
from transformers import (
    AutoBackbone,
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    get_cosine_schedule_with_warmup,
)

try:
    import trackio  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    trackio = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    image_path: Path
    label_path: Path
    image_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen DINOv3 + DETR on YOLO data.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/awiros-tech/Projects/datasets/crowd_human_mot_dataset",
        help="Dataset root containing train/ and val/ folders.",
    )
    parser.add_argument("--phase", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument(
        "--backbone_name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--image_processor_name",
        type=str,
        default="facebook/detr-resnet-50",
    )
    parser.add_argument(
        "--detr_init_model",
        type=str,
        default="facebook/detr-resnet-50",
        help="DETR checkpoint used to initialize encoder/decoder/query/head weights.",
    )
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--shortest_edge", type=int, default=960)
    parser.add_argument("--longest_edge", type=int, default=1536)
    parser.add_argument("--pad_height", type=int, default=960)
    parser.add_argument("--pad_width", type=int, default=1536)

    parser.add_argument("--num_queries", type=int, default=300)
    parser.add_argument("--encoder_layers", type=int, default=6)
    parser.add_argument("--decoder_layers", type=int, default=6)

    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "no"], default="bf16")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=8)

    parser.add_argument("--disable_trackio", action="store_true")
    parser.add_argument("--trackio_project", type=str, default="dinov3-detr-crowdhuman")
    parser.add_argument("--trackio_space_id", type=str, default=None)
    parser.add_argument("--disable_eval_viz", action="store_true")
    parser.add_argument("--eval_viz_max_images", type=int, default=4)
    parser.add_argument("--eval_viz_score_threshold", type=float, default=0.3)

    return parser.parse_args()


def load_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml at {data_yaml}")

    parsed = yaml.safe_load(data_yaml.read_text())
    names = parsed.get("names")
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names, key=lambda x: int(x))]
        return [str(n) for n in ordered]

    raise ValueError("Unsupported names format in data.yaml")


def collect_split_samples(
    dataset_root: Path,
    split: str,
    max_samples: int | None,
    seed: int,
) -> list[Sample]:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    image_paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    image_paths.sort()

    if max_samples is not None and max_samples > 0 and len(image_paths) > max_samples:
        rng = random.Random(seed + (0 if split == "train" else 1))
        rng.shuffle(image_paths)
        image_paths = sorted(image_paths[:max_samples])

    samples: list[Sample] = []
    for idx, image_path in enumerate(image_paths):
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label_dir / f"{image_path.stem}.txt",
                image_id=idx,
            )
        )
    return samples


def load_yolo_labels(label_path: Path, num_classes: int) -> tuple[list[list[float]], list[int]]:
    if not label_path.exists():
        return [], []

    bboxes: list[list[float]] = []
    class_ids: list[int] = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue

        if class_id < 0 or class_id >= num_classes:
            continue
        if bw <= 0.0 or bh <= 0.0:
            continue

        # Clamp to YOLO normalized bounds.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 1e-8), 1.0)
        bh = min(max(bh, 1e-8), 1.0)

        bboxes.append([cx, cy, bw, bh])
        class_ids.append(class_id)
    return bboxes, class_ids


class YoloDetrDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_processor: DetrImageProcessor,
        num_classes: int,
        train: bool,
        seed: int,
    ):
        self.samples = samples
        self.image_processor = image_processor
        self.num_classes = num_classes
        self.train = train
        self.seed = seed

        self.train_augment = None
        if self.train:
            self.train_augment = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        rotate=(-5, 5),
                        shear={"x": (-3, 3), "y": (-3, 3)},
                        translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                        scale=(0.9, 1.1),
                        p=0.4,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1,
                        contrast_limit=0.1,
                        p=0.2,
                    ),
                ],
                #Doubt
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=["class_labels"],
                    min_area=0.0,
                    min_visibility=0.0,
                    clip=True,
                    check_each_transform=False,
                ),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image_np = np.array(image)

        bboxes, class_labels = load_yolo_labels(sample.label_path, self.num_classes)

        if self.train_augment is not None:
            # Albumentations uses NumPy image + YOLO-format bboxes.
            transformed = self.train_augment(
                image=image_np,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            image_np = transformed["image"]
            bboxes = []
            for bbox in transformed["bboxes"]:
                cx, cy, bw, bh = map(float, bbox)
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 1e-8), 1.0)
                bh = min(max(bh, 1e-8), 1.0)
                bboxes.append([cx, cy, bw, bh])
            class_labels = [int(c) for c in transformed["class_labels"]]

        image = Image.fromarray(image_np)
        image_h, image_w = image_np.shape[:2]

        annotations: list[dict[str, Any]] = []
        for class_id, (cx, cy, bw, bh) in zip(class_labels, bboxes):
            x0 = (cx - (bw / 2.0)) * image_w
            y0 = (cy - (bh / 2.0)) * image_h
            w_abs = bw * image_w
            h_abs = bh * image_h

            x1 = max(0.0, min(float(image_w), x0))
            y1 = max(0.0, min(float(image_h), y0))
            x2 = max(0.0, min(float(image_w), x0 + w_abs))
            y2 = max(0.0, min(float(image_h), y0 + h_abs))

            final_w = x2 - x1
            final_h = y2 - y1
            if final_w <= 0.0 or final_h <= 0.0:
                continue

            annotations.append(
                {
                    "bbox": [x1, y1, final_w, final_h],
                    "category_id": int(class_id),
                    "area": float(final_w * final_h),
                    "iscrowd": 0,
                }
            )

        target = {"image_id": sample.image_id, "annotations": annotations}
        encoded = self.image_processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
        }


def collate_fn(
    batch: list[dict[str, Any]],
    image_processor: DetrImageProcessor,
) -> dict[str, Any]:
    # Dynamic padding per batch avoids fixed-pad failures on tall portrait images.
    pixel_values = [x["pixel_values"] for x in batch]
    encoding = image_processor.pad(images=pixel_values, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": [x["labels"] for x in batch],
    }


def move_to_device(item: Any, device: torch.device) -> Any:
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, dict):
        return {k: move_to_device(v, device) for k, v in item.items()}
    if isinstance(item, list):
        return [move_to_device(v, device) for v in item]
    if isinstance(item, tuple):
        return tuple(move_to_device(v, device) for v in item)
    return item


def tensor_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float("nan")
        return float(value.detach().cpu().item())
    return float(value)


def flatten_map_metrics(raw_metrics: dict[str, Any], id2label: dict[int, str]) -> dict[str, float]:
    keys = [
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    ]
    output = {f"eval_{k}": tensor_to_float(raw_metrics[k]) for k in keys if k in raw_metrics}

    classes = raw_metrics.get("classes")
    map_per_class = raw_metrics.get("map_per_class")
    mar_100_per_class = raw_metrics.get("mar_100_per_class")

    if isinstance(classes, torch.Tensor) and isinstance(map_per_class, torch.Tensor):
        for class_idx, map_value in zip(classes.tolist(), map_per_class.tolist()):
            class_name = id2label.get(int(class_idx), str(int(class_idx)))
            output[f"eval_map_{class_name}"] = float(map_value)

    if isinstance(classes, torch.Tensor) and isinstance(mar_100_per_class, torch.Tensor):
        for class_idx, mar_value in zip(classes.tolist(), mar_100_per_class.tolist()):
            class_name = id2label.get(int(class_idx), str(int(class_idx)))
            output[f"eval_mar_100_{class_name}"] = float(mar_value)

    return output


@torch.no_grad()
def evaluate_epoch(
    model: DetrForObjectDetection,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    image_processor: DetrImageProcessor,
    id2label: dict[int, str],
) -> dict[str, float]:
    if accelerator.num_processes != 1:
        raise RuntimeError("This script currently supports single-process evaluation only.")

    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    metric.warn_on_many_detections = False

    eval_loss = 0.0
    num_batches = 0

    for batch in eval_dataloader:
        batch = move_to_device(batch, accelerator.device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            labels=batch["labels"],
        )

        eval_loss += outputs.loss.detach().float().item()
        num_batches += 1

        target_sizes = torch.stack([label["orig_size"] for label in batch["labels"]])
        detections = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.0,
            target_sizes=target_sizes,
        )

        preds: list[dict[str, torch.Tensor]] = []
        targets: list[dict[str, torch.Tensor]] = []
        for detection, label in zip(detections, batch["labels"]):
            preds.append(
                {
                    "boxes": detection["boxes"].detach().cpu(),
                    "scores": detection["scores"].detach().cpu(),
                    "labels": detection["labels"].detach().cpu(),
                }
            )

            gt_boxes = label["boxes"]  # normalized cxcywh
            orig_h, orig_w = label["orig_size"]
            scale = torch.tensor(
                [orig_w, orig_h, orig_w, orig_h],
                dtype=gt_boxes.dtype,
                device=gt_boxes.device,
            )
            gt_boxes_abs = gt_boxes * scale
            gt_boxes_xyxy = box_convert(gt_boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
            targets.append(
                {
                    "boxes": gt_boxes_xyxy.detach().cpu(),
                    "labels": label["class_labels"].detach().cpu(),
                }
            )

        metric.update(preds, targets)

    raw = metric.compute()
    metrics = flatten_map_metrics(raw, id2label)
    metrics["eval_loss"] = eval_loss / max(1, num_batches)
    return metrics


def _pixel_values_to_pil_image(
    pixel_values: torch.Tensor,
    image_mean: list[float] | tuple[float, ...],
    image_std: list[float] | tuple[float, ...],
) -> Image.Image:
    image = pixel_values.detach().cpu().float().permute(1, 2, 0).numpy()
    mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
    image = (image * std) + mean
    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255.0).astype(np.uint8)
    return Image.fromarray(image_uint8)


def _draw_xyxy_boxes(
    image: Image.Image,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    id2label: dict[int, str],
    color: tuple[int, int, int],
    prefix: str,
    scores: torch.Tensor | None = None,
) -> None:
    draw = ImageDraw.Draw(image)
    for idx in range(boxes_xyxy.shape[0]):
        x0, y0, x1, y1 = boxes_xyxy[idx].tolist()
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        class_id = int(labels[idx].item())
        class_name = id2label.get(class_id, str(class_id))
        if scores is None:
            text = f"{prefix}:{class_name}"
        else:
            text = f"{prefix}:{class_name} {float(scores[idx].item()):.2f}"
        text_x = max(0, int(x0))
        text_y = max(0, int(y0) - 12)
        draw.text((text_x, text_y), text, fill=color)


@torch.no_grad()
def save_eval_batch_visualization(
    model: DetrForObjectDetection,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    image_processor: DetrImageProcessor,
    id2label: dict[int, str],
    run_dir: Path,
    epoch: int,
    score_threshold: float,
    max_images: int,
) -> list[Path]:
    if not accelerator.is_main_process:
        return []

    try:
        raw_batch = next(iter(eval_dataloader))
    except StopIteration:
        return []

    batch = move_to_device(raw_batch, accelerator.device)
    model.eval()
    outputs = model(
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
    )
    target_sizes = torch.stack([label["size"] for label in batch["labels"]])
    detections = image_processor.post_process_object_detection(
        outputs=outputs,
        threshold=score_threshold,
        target_sizes=target_sizes,
    )

    image_mean = tuple(float(v) for v in image_processor.image_mean)
    image_std = tuple(float(v) for v in image_processor.image_std)
    num_to_save = min(max_images, len(detections), len(raw_batch["labels"]), raw_batch["pixel_values"].shape[0])
    if num_to_save <= 0:
        return []

    viz_dir = run_dir / "eval_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx in range(num_to_save):
        image = _pixel_values_to_pil_image(raw_batch["pixel_values"][idx], image_mean, image_std)

        gt = raw_batch["labels"][idx]
        size_h, size_w = gt["size"].tolist()
        gt_scale = torch.tensor(
            [size_w, size_h, size_w, size_h],
            dtype=gt["boxes"].dtype,
            device=gt["boxes"].device,
        )
        gt_boxes_xyxy = box_convert(gt["boxes"] * gt_scale, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu()
        gt_labels = gt["class_labels"].detach().cpu()
        _draw_xyxy_boxes(
            image=image,
            boxes_xyxy=gt_boxes_xyxy,
            labels=gt_labels,
            id2label=id2label,
            color=(0, 255, 0),
            prefix="GT",
        )

        pred = detections[idx]
        pred_boxes = pred["boxes"].detach().cpu()
        pred_labels = pred["labels"].detach().cpu()
        pred_scores = pred["scores"].detach().cpu()
        _draw_xyxy_boxes(
            image=image,
            boxes_xyxy=pred_boxes,
            labels=pred_labels,
            id2label=id2label,
            color=(255, 0, 0),
            prefix="P",
            scores=pred_scores,
        )

        out_path = viz_dir / f"epoch-{epoch:03d}-sample-{idx:02d}.jpg"
        image.save(out_path, quality=95)
        saved_paths.append(out_path)

    return saved_paths


def build_model(
    backbone_name: str,
    detr_init_model: str,
    id2label: dict[int, str],
    label2id: dict[str, int],
    num_queries: int,
    encoder_layers: int,
    decoder_layers: int,
    local_files_only: bool,
) -> DetrForObjectDetection:
    backbone = AutoBackbone.from_pretrained(
        backbone_name,
        local_files_only=local_files_only,
    )

    # Start from COCO-pretrained DETR weights, while remapping label head to current dataset classes.
    model = DetrForObjectDetection.from_pretrained(
        detr_init_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        num_queries=num_queries,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )

    model.config.auxiliary_loss = True
    model.config.backbone_config = backbone.config.to_dict()
    model.config.use_timm_backbone = False
    model.model.config.auxiliary_loss = True
    model.model.config.backbone_config = backbone.config.to_dict()
    model.model.config.use_timm_backbone = False

    # Inject pretrained DINOv3 backbone, then adapt projection channels.
    model.model.backbone.model = backbone
    model.model.backbone.intermediate_channel_sizes = backbone.channels
    model.model.input_projection = torch.nn.Conv2d(backbone.channels[-1], model.config.d_model, kernel_size=1)
    model.model.freeze_backbone()
    return model


def create_run_dir(base_output_dir: Path, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if run_name is None:
        run_name = f"dinov3-detr-{timestamp}"
    return base_output_dir / run_name


def prune_checkpoints(run_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted([p for p in run_dir.glob("checkpoint-epoch-*") if p.is_dir()])
    if len(checkpoints) <= keep_last:
        return
    for old_ckpt in checkpoints[: len(checkpoints) - keep_last]:
        for child in old_ckpt.rglob("*"):
            if child.is_file():
                child.unlink()
        for child in sorted(old_ckpt.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        if old_ckpt.exists():
            old_ckpt.rmdir()


def save_checkpoint(
    accelerator: Accelerator,
    model: DetrForObjectDetection,
    image_processor: DetrImageProcessor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_dir: Path,
    epoch: int,
    metrics: dict[str, float],
    save_total_limit: int,
) -> None:
    checkpoint_dir = run_dir / f"checkpoint-epoch-{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        checkpoint_dir,
        state_dict=state_dict,
        save_function=accelerator.save,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        image_processor.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "metrics": metrics,
            },
            checkpoint_dir / "training_state.pt",
        )
        (checkpoint_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        prune_checkpoints(run_dir, save_total_limit)


def save_best_model(
    accelerator: Accelerator,
    model: DetrForObjectDetection,
    image_processor: DetrImageProcessor,
    run_dir: Path,
    metrics: dict[str, float],
) -> None:
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped.save_pretrained(
        best_dir,
        state_dict=state_dict,
        save_function=accelerator.save,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        image_processor.save_pretrained(best_dir)
        (best_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.phase == "pilot":
        if args.max_train_samples is None:
            args.max_train_samples = 10
        if args.max_val_samples is None:
            args.max_val_samples = 10

    if not args.disable_trackio and trackio is None:
        raise RuntimeError(
            "Trackio is not available. Install it (`pip install trackio`) or run with --disable_trackio."
        )

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    run_dir = create_run_dir(output_dir, args.run_name)

    class_names = load_class_names(dataset_root)
    id2label = {idx: name for idx, name in enumerate(class_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    train_samples = collect_split_samples(
        dataset_root=dataset_root,
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    eval_samples = collect_split_samples(
        dataset_root=dataset_root,
        split="val",
        max_samples=args.max_val_samples,
        seed=args.seed,
    )

    image_processor = DetrImageProcessor.from_pretrained(
        args.image_processor_name,
        size={"shortest_edge": args.shortest_edge, "longest_edge": args.longest_edge},
        do_pad=False,
        local_files_only=args.local_files_only,
    )

    collate = partial(collate_fn, image_processor=image_processor)

    train_dataset = YoloDetrDataset(
        samples=train_samples,
        image_processor=image_processor,
        num_classes=len(class_names),
        train=True,
        seed=args.seed,
    )
    eval_dataset = YoloDetrDataset(
        samples=eval_samples,
        image_processor=image_processor,
        num_classes=len(class_names),
        train=False,
        seed=args.seed + 1,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(
        backbone_name=args.backbone_name,
        detr_init_model=args.detr_init_model,
        id2label=id2label,
        label2id=label2id,
        num_queries=args.num_queries,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        local_files_only=args.local_files_only,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision,
    )

    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
    )

    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        config_dump = vars(args).copy()
        config_dump["class_names"] = class_names
        config_dump["num_train_samples"] = len(train_samples)
        config_dump["num_eval_samples"] = len(eval_samples)
        (run_dir / "run_config.json").write_text(json.dumps(config_dump, indent=2))

    if accelerator.is_main_process and not args.disable_trackio:
        trackio_kwargs = {
            "project": args.trackio_project,
            "name": run_dir.name,
            "config": {
                "phase": args.phase,
                "backbone_name": args.backbone_name,
                "num_queries": args.num_queries,
                "encoder_layers": args.encoder_layers,
                "decoder_layers": args.decoder_layers,
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "grad_accum_steps": args.gradient_accumulation_steps,
                "epochs": args.num_train_epochs,
                "train_samples": len(train_samples),
                "eval_samples": len(eval_samples),
            },
        }
        if args.trackio_space_id:
            trackio_kwargs["space_id"] = args.trackio_space_id
        trackio.init(**trackio_kwargs)

    best_map = float("-inf")
    history_path = run_dir / "metrics_history.jsonl"
    global_step = 0

    try:
        for epoch in range(1, args.num_train_epochs + 1):
            model.train()
            running_loss = 0.0
            step_count = 0
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch:02d}/{args.num_train_epochs:02d}",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                leave=True,
            )

            for step, batch in enumerate(train_dataloader, start=1):
                step_count += 1
                batch = move_to_device(batch, accelerator.device)

                with accelerator.accumulate(model):
                    outputs = model(
                        pixel_values=batch["pixel_values"],
                        pixel_mask=batch["pixel_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                gathered_loss = accelerator.gather_for_metrics(loss.detach().unsqueeze(0)).mean().item()
                running_loss += gathered_loss

                if accelerator.is_main_process and not args.disable_trackio:
                    if global_step % args.log_every_steps == 0:
                        trackio.log(
                            {
                                "global_step": global_step,
                                "epoch_progress": epoch - 1 + (step / max(1, len(train_dataloader))),
                                "train_loss_step": gathered_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            }
                        )
                if accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=f"{gathered_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )
                    tqdm.write(
                        f"[Epoch {epoch:02d} | Step {step:05d}/{len(train_dataloader):05d}] "
                        f"loss={gathered_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )

                global_step += 1

            if accelerator.is_main_process:
                progress_bar.close()
            train_loss = running_loss / max(1, step_count)
            accelerator.wait_for_everyone()

            eval_metrics = evaluate_epoch(
                model=model,
                eval_dataloader=eval_dataloader,
                accelerator=accelerator,
                image_processor=image_processor,
                id2label=id2label,
            )

            viz_paths: list[Path] = []
            if not args.disable_eval_viz:
                viz_paths = save_eval_batch_visualization(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    accelerator=accelerator,
                    image_processor=image_processor,
                    id2label=id2label,
                    run_dir=run_dir,
                    epoch=epoch,
                    score_threshold=args.eval_viz_score_threshold,
                    max_images=args.eval_viz_max_images,
                )

            epoch_metrics: dict[str, float] = {"epoch": float(epoch), "train_loss": float(train_loss)}
            epoch_metrics.update(eval_metrics)

            if accelerator.is_main_process:
                with history_path.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(epoch_metrics) + "\n")

                if not args.disable_trackio:
                    trackio.log(epoch_metrics)

            save_checkpoint(
                accelerator=accelerator,
                model=model,
                image_processor=image_processor,
                optimizer=optimizer,
                scheduler=scheduler,
                run_dir=run_dir,
                epoch=epoch,
                metrics=epoch_metrics,
                save_total_limit=args.save_total_limit,
            )

            current_map = epoch_metrics.get("eval_map", float("nan"))
            if math.isfinite(current_map) and current_map > best_map:
                best_map = current_map
                save_best_model(
                    accelerator=accelerator,
                    model=model,
                    image_processor=image_processor,
                    run_dir=run_dir,
                    metrics=epoch_metrics,
                )

            if accelerator.is_main_process:
                print(
                    f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
                    f"eval_map={epoch_metrics.get('eval_map', float('nan')):.4f} "
                    f"eval_map_50={epoch_metrics.get('eval_map_50', float('nan')):.4f} "
                    f"eval_mar_100={epoch_metrics.get('eval_mar_100', float('nan')):.4f}"
                )
                if viz_paths:
                    print(f"[Epoch {epoch:02d}] saved eval visualizations to: {viz_paths[0].parent}")

    finally:
        if accelerator.is_main_process and not args.disable_trackio:
            trackio.finish()

    if accelerator.is_main_process:
        print(f"Training finished. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
