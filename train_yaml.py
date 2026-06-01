"""
AURA-NET - YOLO-style Training with YAML Config - FIXED VERSION
===============================================================

Fixes:
- Robust Kaggle/Roboflow YOLO data.yaml path resolving
- Supports train/valid/val/test structures
- Reads jpg/jpeg/png/bmp/webp/tif/tiff with upper/lowercase
- Avoids overwriting PyTorch model.train()
- Better dataset checks
- Uses num_classes from data.yaml
- Computes mAP50 and mAP50-95 more properly than mAP50 * 0.7
- Supports all AURA-NET variants (Nano, Tiny, S, M, L, X)

Usage:
    # Import any variant
    from models import AURANetTiny  # or AURANet_Tiny, AURANetS, etc.
    from train_yaml import add_train_method

    # Add train method to all variants
    add_train_method()

    # Create model
    model = AURANetTiny(num_classes=9)

    # Train
    model.train_model(
        data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
        epochs=200,
        batch=8,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        lr0=1e-4,
        patience=60,
        seed=42,
        project="runs/train",
        name="auranet_tiny_bccd",
        workers=4
    )
"""

import os
import sys
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import random
import time
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from losses.aura_v3_loss import AURANetLoss


IMG_EXTS = [
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF"
]


class YOLODataset(Dataset):
    """YOLO-format dataset loader: class_id cx cy w h"""

    def __init__(self, img_dir, label_dir, img_size=640, augment=False, num_classes=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.num_classes = num_classes

        self.img_files = []
        for ext in IMG_EXTS:
            self.img_files.extend(list(self.img_dir.glob(f"*{ext}")))
        self.img_files = sorted(set(self.img_files))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        label_path = self.label_dir / f"{img_path.stem}.txt"
        boxes = self.load_labels(label_path)

        if self.augment:
            image, boxes = self.apply_augmentation(image, boxes)

        return image, boxes

    def load_labels(self, label_path):
        boxes = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(
                            f"Invalid label format in {label_path}, line {line_idx}. "
                            f"Expected 5 values: class_id cx cy w h, got {len(parts)}."
                        )

                    class_id, cx, cy, w, h = map(float, parts)

                    if self.num_classes is not None:
                        if int(class_id) < 0 or int(class_id) >= self.num_classes:
                            raise ValueError(
                                f"Class id {int(class_id)} out of range in {label_path}, line {line_idx}. "
                                f"Dataset has nc={self.num_classes}, valid ids are 0..{self.num_classes - 1}."
                            )

                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        raise ValueError(
                            f"Invalid YOLO bbox in {label_path}, line {line_idx}: "
                            f"{cx} {cy} {w} {h}. Values must be normalized between 0 and 1."
                        )

                    boxes.append([cx, cy, w, h, class_id])

        if len(boxes) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(boxes, dtype=torch.float32)

    def apply_augmentation(self, image, boxes):
        # Horizontal flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[2])
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]
        return image, boxes


def collate_fn(batch):
    images = []
    targets = []

    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)

    images = torch.stack(images, dim=0)
    return images, targets


def load_yaml_config(yaml_path):
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty YAML file: {yaml_path}")

    required = ["train", "val", "nc", "names"]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Missing keys in data.yaml: {missing}")

    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def normalize_names(names):
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    return list(names)


def find_existing_dir(candidates, name):
    for c in candidates:
        c = Path(c).resolve()
        if c.exists() and c.is_dir():
            return c

    msg = f"Cannot find {name} directory. Tried:\n"
    for c in candidates:
        msg += f"  - {Path(c).resolve()}\n"
    raise FileNotFoundError(msg)


def resolve_yolo_split_path(data_yaml, data_config, split_key):
    """
    Robust resolver for Roboflow/Kaggle YOLO paths.

    Handles:
    path: /root/dataset
    train: train/images
    val: valid/images

    Also handles problematic:
    path: .
    train: ../train/images
    val: ../valid/images

    On Kaggle, if data.yaml is inside /bccd-new, we prefer resolving relative to the YAML directory.
    """

    data_yaml = Path(data_yaml).resolve()
    yaml_dir = data_yaml.parent

    path_value = data_config.get("path", None)
    split_value = data_config[split_key]

    split_path = Path(split_value)

    candidates = []

    # Absolute split path
    if split_path.is_absolute():
        candidates.append(split_path)

    # YAML directory + split as written
    candidates.append(yaml_dir / split_path)

    # If split starts with ../, also try removing ../ and resolving inside yaml_dir
    clean_parts = [p for p in split_path.parts if p not in ("..", ".")]
    if clean_parts:
        candidates.append(yaml_dir.joinpath(*clean_parts))

    # Root from data_config["path"]
    if path_value:
        path_obj = Path(path_value)

        if path_obj.is_absolute():
            data_root = path_obj
        elif str(path_obj) in [".", "./"]:
            data_root = yaml_dir
        else:
            # Most reliable for Kaggle/Roboflow:
            # relative path should be relative to YAML dir, not current notebook dir.
            data_root = yaml_dir / path_obj

        candidates.append(data_root / split_path)

        if clean_parts:
            candidates.append(data_root.joinpath(*clean_parts))

    # Common split aliases
    # If val points to valid/images, also try val/images, and vice versa.
    split_str = str(split_value).replace("\\", "/")
    if "valid/images" in split_str:
        candidates.append(yaml_dir / split_str.replace("valid/images", "val/images"))
    if "val/images" in split_str:
        candidates.append(yaml_dir / split_str.replace("val/images", "valid/images"))

    # If split does not include images, try split/images
    if "images" not in split_str:
        candidates.append(yaml_dir / split_value / "images")
        if path_value:
            candidates.append(data_root / split_value / "images")

    img_dir = find_existing_dir(candidates, f"{split_key} images")

    # Label dir from image dir
    if "images" in img_dir.parts:
        parts = list(img_dir.parts)
        last_images_idx = len(parts) - 1 - parts[::-1].index("images")
        parts[last_images_idx] = "labels"
        label_dir = Path(*parts)
    else:
        label_dir = img_dir.parent / "labels"

    if not label_dir.exists():
        # Try common alternatives
        alt_candidates = [
            img_dir.parent.parent / "labels" / img_dir.parent.name,
            img_dir.parent / "labels",
            yaml_dir / str(split_value).replace("images", "labels"),
        ]
        label_dir = find_existing_dir(alt_candidates, f"{split_key} labels")

    return img_dir.resolve(), label_dir.resolve()


def print_training_header(config, data_config, device, save_dir):
    names = normalize_names(data_config["names"])

    print("\n" + "=" * 100)
    print(f"{'AURA-NET Training':^100}")
    print("=" * 100)

    print(f"\n{'Configuration':<30} {'Value':<70}")
    print("-" * 100)
    print(f"{'Device':<30} {str(device):<70}")
    print(f"{'Epochs':<30} {config['epochs']:<70}")
    print(f"{'Batch Size':<30} {config['batch']:<70}")
    print(f"{'Image Size':<30} {config['imgsz']:<70}")
    print(f"{'Optimizer':<30} {config['optimizer']:<70}")
    print(f"{'Learning Rate':<30} {config.get('lr0', 1e-4):<70}")
    print(f"{'Patience':<30} {config['patience']:<70}")
    print(f"{'Seed':<30} {config['seed']:<70}")
    print(f"{'Dataset Path':<30} {data_config.get('path', 'N/A'):<70}")
    print(f"{'Classes':<30} {data_config['nc']:<70}")
    print(f"{'Class Names':<30} {', '.join(names):<70}")
    print(f"{'Save Directory':<30} {str(save_dir):<70}")
    print("=" * 100)


def print_model_summary(model, img_size, device):
    print("\n" + "=" * 100)
    print(f"{'Model Summary':^100}")
    print("=" * 100)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'Metric':<30} {'Value':<70}")
    print("-" * 100)
    print(f"{'Total Parameters':<30} {total_params:>12,}")
    print(f"{'Trainable Parameters':<30} {trainable_params:>12,}")
    print(f"{'Parameters (M)':<30} {total_params / 1e6:>12.2f}")

    try:
        from thop import profile, clever_format
        x = torch.randn(1, 3, img_size, img_size).to(device)
        model.eval()
        flops, _ = profile(model, inputs=(x,), verbose=False)
        flops, _ = clever_format([flops, total_params], "%.3f")
        print(f"{'GFLOPs':<30} {flops:>12}")
    except Exception as e:
        print(f"{'GFLOPs':<30} {'N/A':>12}")
        print(f"{'GFLOPs Error':<30} {str(e)[:70]:<70}")

    print("=" * 100)


def box_cxcywh_to_xyxy(boxes):
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))

    xyxy = torch.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def box_iou(boxes1, boxes2):
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    area1 = torch.clamp(boxes1[:, 2] - boxes1[:, 0], min=0) * torch.clamp(boxes1[:, 3] - boxes1[:, 1], min=0)
    area2 = torch.clamp(boxes2[:, 2] - boxes2[:, 0], min=0) * torch.clamp(boxes2[:, 3] - boxes2[:, 1], min=0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def compute_ap(recall, precision):
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([1.0]), precision, torch.tensor([0.0])])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap.item()


def compute_metrics(predictions, targets, num_classes):
    """
    predictions format expected:
        pred["boxes"]     : Tensor[N,4] cxcywh normalized
        pred["scores"]    : Tensor[N]
        pred["class_ids"] : Tensor[N]

    targets:
        Tensor[M,5] cx cy w h class_id
    """

    iou_thresholds = torch.arange(0.50, 0.96, 0.05)

    aps = []
    aps50 = []

    total_tp_50 = torch.zeros(num_classes)
    total_fp_50 = torch.zeros(num_classes)
    total_fn_50 = torch.zeros(num_classes)

    for cls in range(num_classes):
        cls_preds = []
        cls_targets_count = 0

        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            target = target.detach().cpu()
            target_cls = target[target[:, 4].long() == cls]
            cls_targets_count += len(target_cls)

            if pred is None or len(pred.get("boxes", [])) == 0:
                continue

            pred_boxes = pred["boxes"].detach().cpu()
            pred_cls = pred["class_ids"].detach().cpu().long()

            if "scores" in pred:
                pred_scores = pred["scores"].detach().cpu()
            elif "conf" in pred:
                pred_scores = pred["conf"].detach().cpu()
            else:
                pred_scores = torch.ones(len(pred_boxes))

            keep = pred_cls == cls
            for b, s in zip(pred_boxes[keep], pred_scores[keep]):
                cls_preds.append((img_idx, float(s), b))

        if cls_targets_count == 0:
            continue

        cls_preds.sort(key=lambda x: x[1], reverse=True)

        ap_per_iou = []

        for iou_thr in iou_thresholds:
            tp = torch.zeros(len(cls_preds))
            fp = torch.zeros(len(cls_preds))
            matched = set()

            for pred_idx, (img_idx, score, pred_box) in enumerate(cls_preds):
                target = targets[img_idx].detach().cpu()
                target_cls = target[target[:, 4].long() == cls]

                if len(target_cls) == 0:
                    fp[pred_idx] = 1
                    continue

                ious = box_iou(pred_box.view(1, 4), target_cls[:, :4]).view(-1)
                best_iou, best_target_idx = ious.max(0)

                match_key = (img_idx, int(best_target_idx))

                if best_iou >= iou_thr and match_key not in matched:
                    tp[pred_idx] = 1
                    matched.add(match_key)
                else:
                    fp[pred_idx] = 1

            if len(cls_preds) == 0:
                ap = 0.0
            else:
                tp_cum = torch.cumsum(tp, 0)
                fp_cum = torch.cumsum(fp, 0)

                recall = tp_cum / (cls_targets_count + 1e-7)
                precision = tp_cum / (tp_cum + fp_cum + 1e-7)

                ap = compute_ap(recall, precision)

            ap_per_iou.append(ap)

            if abs(float(iou_thr) - 0.50) < 1e-6:
                aps50.append(ap)

                # For simple P/R at IoU 0.50
                total_tp_50[cls] = tp.sum() if len(cls_preds) else 0
                total_fp_50[cls] = fp.sum() if len(cls_preds) else 0
                total_fn_50[cls] = max(cls_targets_count - total_tp_50[cls], 0)

        aps.append(float(np.mean(ap_per_iou)))

    precision_per_class = total_tp_50 / (total_tp_50 + total_fp_50 + 1e-7)
    recall_per_class = total_tp_50 / (total_tp_50 + total_fn_50 + 1e-7)

    precision = precision_per_class.mean().item()
    recall = recall_per_class.mean().item()
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)

    map50 = float(np.mean(aps50)) if len(aps50) else 0.0
    map50_95 = float(np.mean(aps)) if len(aps) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": map50,
        "mAP50-95": map50_95,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    losses = defaultdict(float)

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}/{total_epochs}",
        bar_format="{l_bar}{bar:10}{r_bar}"
    )

    for batch_idx, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        for k, v in loss_dict.items():
            losses[k] += float(v.detach().item())

        if batch_idx % 10 == 0:
            pbar.set_postfix({
                "loss": f"{losses['loss'] / (batch_idx + 1):.4f}",
                "box": f"{losses.get('loss_box', 0.0) / (batch_idx + 1):.4f}",
                "cls": f"{losses.get('loss_class', 0.0) / (batch_idx + 1):.4f}",
            })

    return {k: v / max(len(dataloader), 1) for k, v in losses.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    losses = defaultdict(float)
    all_predictions = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Validating", bar_format="{l_bar}{bar:10}{r_bar}")

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets_device = [t.to(device, non_blocking=True) for t in targets]

        outputs = model(images)
        loss_dict = criterion(outputs, targets_device)

        if hasattr(model, "predict"):
            predictions = model.predict(images, conf_threshold=0.25)
        else:
            raise AttributeError("Your AURANet model must implement model.predict(images, conf_threshold=0.25).")

        all_predictions.extend(predictions)
        all_targets.extend([t.detach().cpu() for t in targets])

        for k, v in loss_dict.items():
            losses[k] += float(v.detach().item())

    avg_losses = {k: v / max(len(dataloader), 1) for k, v in losses.items()}
    metrics = compute_metrics(all_predictions, all_targets, num_classes)

    return avg_losses, metrics


def print_epoch_results(epoch, total_epochs, train_losses, val_losses, val_metrics, epoch_time, lr):
    print("\n" + "=" * 100)
    print(f"Epoch {epoch}/{total_epochs} - Time: {epoch_time:.1f}s - LR: {lr:.8f}")
    print("=" * 100)

    print(f"{'Metric':<20} {'Train':<15} {'Val':<15}")
    print("-" * 100)
    print(f"{'Loss':<20} {train_losses.get('loss', 0):>14.4f} {val_losses.get('loss', 0):>14.4f}")
    print(f"{'Box Loss':<20} {train_losses.get('loss_box', 0):>14.4f} {val_losses.get('loss_box', 0):>14.4f}")
    print(f"{'Class Loss':<20} {train_losses.get('loss_class', 0):>14.4f} {val_losses.get('loss_class', 0):>14.4f}")
    print("-" * 100)
    print(f"{'Precision (P)':<20} {'-':>14} {val_metrics['precision']:>14.4f}")
    print(f"{'Recall (R)':<20} {'-':>14} {val_metrics['recall']:>14.4f}")
    print(f"{'F1':<20} {'-':>14} {val_metrics['f1']:>14.4f}")
    print(f"{'mAP@0.5':<20} {'-':>14} {val_metrics['mAP50']:>14.4f}")
    print(f"{'mAP@0.5:0.95':<20} {'-':>14} {val_metrics['mAP50-95']:>14.4f}")
    print("=" * 100 + "\n")


def train_model(
    model,
    data,
    epochs=100,
    batch=16,
    imgsz=640,
    device=0,
    optimizer="AdamW",
    lr0=1e-4,
    weight_decay=0.0001,
    patience=50,
    seed=42,
    cache=False,
    project="runs/train",
    name="exp",
    workers=2,
    **kwargs
):
    set_seed(seed)

    if device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    data = Path(data).resolve()
    data_config = load_yaml_config(data)
    data_config["names"] = normalize_names(data_config["names"])

    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
        "lr0": lr0,
        "weight_decay": weight_decay,
        "patience": patience,
        "seed": seed,
    }

    print_training_header(config, data_config, device, save_dir)

    print("\nLoading datasets...")

    train_img_dir, train_label_dir = resolve_yolo_split_path(data, data_config, "train")
    val_img_dir, val_label_dir = resolve_yolo_split_path(data, data_config, "val")

    print(f"  YAML file:     {data}")
    print(f"  Train images:  {train_img_dir}")
    print(f"  Train labels:  {train_label_dir}")
    print(f"  Val images:    {val_img_dir}")
    print(f"  Val labels:    {val_label_dir}")

    nc = int(data_config["nc"])

    train_dataset = YOLODataset(
        train_img_dir,
        train_label_dir,
        img_size=imgsz,
        augment=True,
        num_classes=nc
    )

    val_dataset = YOLODataset(
        val_img_dir,
        val_label_dir,
        img_size=imgsz,
        augment=False,
        num_classes=nc
    )

    print(f"\n  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training images found in {train_img_dir}.\n"
            f"Supported extensions: {IMG_EXTS}\n"
            f"Check your data.yaml and Kaggle dataset folder structure."
        )

    if len(val_dataset) == 0:
        raise ValueError(
            f"No validation images found in {val_img_dir}.\n"
            f"Supported extensions: {IMG_EXTS}\n"
            f"Check your data.yaml and Kaggle dataset folder structure."
        )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = model.to(device)

    print_model_summary(model, imgsz, device)

    criterion = AURANetLoss(num_classes=nc)

    opt_name = str(optimizer).lower()

    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)
    elif opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}. Use AdamW, Adam, or SGD.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs,
        eta_min=lr0 * 0.01
    )

    with open(save_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                **config,
                "data": str(data),
                "data_config": data_config,
                "train_images": str(train_img_dir),
                "val_images": str(val_img_dir),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print("\n" + "=" * 100)
    print("Starting training...")
    print("=" * 100 + "\n")

    best_map50 = -1.0
    patience_counter = 0

    results_csv = save_dir / "results.csv"
    with open(results_csv, "w", encoding="utf-8") as f:
        f.write(
            "epoch,lr,train_loss,train_box_loss,train_cls_loss,"
            "val_loss,val_box_loss,val_cls_loss,precision,recall,f1,map50,map50_95\n"
        )

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_losses = train_one_epoch(
            model,
            train_loader,
            criterion,
            opt,
            device,
            epoch,
            epochs
        )

        val_losses, val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            nc
        )

        scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print_epoch_results(
            epoch,
            epochs,
            train_losses,
            val_losses,
            val_metrics,
            epoch_time,
            current_lr
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {
                "train": train_losses,
                "val": val_losses,
                "metrics": val_metrics,
            },
            "data_config": data_config,
            "num_classes": nc,
            "class_names": data_config["names"],
        }

        torch.save(checkpoint, save_dir / "last.pt")

        if val_metrics["mAP50"] > best_map50:
            best_map50 = val_metrics["mAP50"]
            torch.save(checkpoint, save_dir / "best.pt")
            print(f"[BEST] New best mAP@0.5: {best_map50:.4f}\n")
            patience_counter = 0
        else:
            patience_counter += 1

        with open(results_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{current_lr},"
                f"{train_losses.get('loss', 0):.6f},"
                f"{train_losses.get('loss_box', 0):.6f},"
                f"{train_losses.get('loss_class', 0):.6f},"
                f"{val_losses.get('loss', 0):.6f},"
                f"{val_losses.get('loss_box', 0):.6f},"
                f"{val_losses.get('loss_class', 0):.6f},"
                f"{val_metrics['precision']:.6f},"
                f"{val_metrics['recall']:.6f},"
                f"{val_metrics['f1']:.6f},"
                f"{val_metrics['mAP50']:.6f},"
                f"{val_metrics['mAP50-95']:.6f}\n"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs. patience={patience}")
            break

    print("\n" + "=" * 100)
    print(f"{'Training Completed!':^100}")
    print("=" * 100)
    print(f"Best mAP@0.5: {best_map50:.4f}")
    print(f"Models saved to: {save_dir}")
    print(f"Results saved to: {results_csv}")
    print("=" * 100 + "\n")

    return save_dir


def add_train_method():
    """
    Add train_model method to all AURA-NET variants.
    Does NOT override PyTorch model.train().
    """
    from models.variants import AURANetBase
    AURANetBase.train_model = train_model


if __name__ == "__main__":
    from models import AURANetTiny

    add_train_method()

    model = AURANetTiny(num_classes=9)

    model.train_model(
        data="/kaggle/input/datasets/kcnttngotruongan/bccd-new/data.yaml",
        epochs=200,
        batch=24,
        imgsz=640,
        device=0,
        optimizer="AdamW",
        lr0=2e-4,
        patience=70,
        seed=42,
        project="runs/train",
        name="auranet_tiny_bccd",
        workers=4,
    )