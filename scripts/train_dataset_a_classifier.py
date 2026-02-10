#!/usr/bin/env python
"""Train a cardiomegaly-vs-healthy classifier on Dataset A (PA-only).

This creates a *new* classifier using the already-exported Dataset A folders:
  - data/raw/healthy/
  - data/raw/cardiomegaly/

Dataset A is PA-only by construction (export script filters View Position == "PA").
Optionally, you can pass a manifest CSV to *enforce* PA-only filtering again.

What it does:
- Loads images from Dataset A
- Stratified train/val/test split (reproducible)
- Fine-tunes an ImageNet-pretrained backbone (default: resnet50)
- Uses mild augmentations suitable for chest X-rays
- Tracks metrics and saves best model by val AUROC (or val F1 if sklearn missing)
- Writes a JSON report with metrics + training history

Outputs:
- Weights: models/classifier/dataset_a_classifier.pt
- Report:  outputs/classifier/dataset_a_report.json

Examples:
  # Fast sanity check (CPU):
  python scripts/train_dataset_a_classifier.py --smoke-test --device cpu

  # Proper training (GPU):
  python scripts/train_dataset_a_classifier.py --device cuda --epochs 30 --batch-size 32

Notes:
- Requires: torch, torchvision. For AUROC/F1/CM extras: scikit-learn.
- On Windows, start with --num-workers 0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def _try_import_sklearn():
    try:
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            roc_auc_score,
        )

        return confusion_matrix, f1_score, roc_auc_score
    except Exception:
        return None


SK = _try_import_sklearn()


@dataclass(frozen=True)
class Split:
    train: List[int]
    val: List[int]
    test: List[int]


class DatasetA(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], image_size: int, train: bool):
        self.items = items
        self.train = train

        # Chest X-rays are grayscale; convert to RGB for ImageNet backbones.
        # Use conservative augments (no flips).
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomRotation(degrees=4),
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.08, contrast=0.08)], p=0.5
                    ),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.tf(img)
        return x, torch.tensor(y, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Dataset A PA-only classifier")

    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV (e.g. data/processed/NIH_A.csv). If provided, enforces PA-only and exact labels.",
    )

    p.add_argument("--model", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"], default="resnet50")
    p.add_argument("--image-size", type=int, default=224)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--early-stop", type=int, default=8, help="Patience in epochs. 0 disables.")

    p.add_argument(
        "--out-weights",
        type=Path,
        default=Path("models") / "classifier" / "dataset_a_classifier.pt",
    )
    p.add_argument(
        "--out-report",
        type=Path,
        default=Path("outputs") / "classifier" / "dataset_a_report.json",
    )

    p.add_argument("--smoke-test", action="store_true", help="Tiny run: 1 epoch, small subset")

    return p.parse_args()


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_items_from_folders(data_root: Path) -> List[Tuple[Path, int]]:
    healthy_dir = data_root / "raw" / "healthy"
    cardio_dir = data_root / "raw" / "cardiomegaly"

    if not healthy_dir.exists() or not cardio_dir.exists():
        raise FileNotFoundError(
            f"Expected Dataset A folders at {healthy_dir} and {cardio_dir}. "
            "(Run your Dataset A export first.)"
        )

    items: List[Tuple[Path, int]] = []
    for p in healthy_dir.iterdir():
        if p.is_file() and _is_image(p):
            items.append((p, 0))
    for p in cardio_dir.iterdir():
        if p.is_file() and _is_image(p):
            items.append((p, 1))

    if not items:
        raise RuntimeError(f"No images found under {data_root / 'raw'}")

    return items


def load_items_from_manifest(data_root: Path, manifest: Path) -> List[Tuple[Path, int]]:
    import pandas as pd

    df = pd.read_csv(manifest)

    required = {"Image Index", "Finding Labels", "View Position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns {sorted(missing)}: {manifest}")

    df = df[df["View Position"] == "PA"].copy()

    labels = df["Finding Labels"].fillna("")
    df = df[(labels == "No Finding") | (labels == "Cardiomegaly")].copy()

    items: List[Tuple[Path, int]] = []
    for _, row in df.iterrows():
        name = str(row["Image Index"])
        label_name = str(row["Finding Labels"]).strip()
        class_name = "healthy" if label_name == "No Finding" else "cardiomegaly"
        y = 0 if class_name == "healthy" else 1

        path = data_root / "raw" / class_name / name
        if path.exists():
            items.append((path, y))

    if not items:
        raise RuntimeError(f"No matching items found in {manifest} that exist under data/raw")

    return items


def stratified_split(y: np.ndarray, train_frac: float, val_frac: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []

    for cls in [0, 1]:
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(1, n - 1)
        n_val = min(max(n_val, 1), n - n_train - 1) if (n - n_train) >= 2 else 0

        train.extend(cls_idx[:n_train].tolist())
        val.extend(cls_idx[n_train : n_train + n_val].tolist())
        test.extend(cls_idx[n_train + n_val :].tolist())

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return Split(train=train, val=val, test=test)


def build_model(name: str) -> nn.Module:
    if name == "resnet18":
        weights = getattr(models, "ResNet18_Weights", None)
        w = weights.IMAGENET1K_V1 if weights else None
        m = models.resnet18(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m

    if name == "resnet34":
        weights = getattr(models, "ResNet34_Weights", None)
        w = weights.IMAGENET1K_V1 if weights else None
        m = models.resnet34(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m

    if name == "resnet50":
        weights = getattr(models, "ResNet50_Weights", None)
        w = weights.IMAGENET1K_V2 if weights else None
        m = models.resnet50(weights=w)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        return m

    if name == "efficientnet_b0":
        weights = getattr(models, "EfficientNet_B0_Weights", None)
        w = weights.IMAGENET1K_V1 if weights else None
        m = models.efficientnet_b0(weights=w)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, 2)
        return m

    raise ValueError(f"Unknown model: {name}")


@torch.no_grad()
def eval_loop(model: nn.Module, loader: DataLoader, device: str, loss_fn: nn.Module) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    ys: List[int] = []
    preds: List[int] = []
    prob1: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))

        p1 = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)

        ys.extend(y.detach().cpu().tolist())
        preds.extend(pred.detach().cpu().tolist())
        prob1.extend(p1.detach().cpu().tolist())

    y_np = np.array(ys, dtype=np.int64)
    pred_np = np.array(preds, dtype=np.int64)
    p1_np = np.array(prob1, dtype=np.float64)

    out: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else math.nan}

    if y_np.size:
        out["acc"] = float((y_np == pred_np).mean())
        tp = int(((pred_np == 1) & (y_np == 1)).sum())
        fp = int(((pred_np == 1) & (y_np == 0)).sum())
        fn = int(((pred_np == 0) & (y_np == 1)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        out["f1"] = float(2 * precision * recall / (precision + recall + 1e-9))
    else:
        out["acc"] = math.nan
        out["f1"] = math.nan

    if SK is not None and y_np.size:
        _, _, roc_auc_score = SK
        try:
            out["auc"] = float(roc_auc_score(y_np, p1_np))
        except Exception:
            out["auc"] = math.nan
    else:
        out["auc"] = math.nan

    return out


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke_test:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 16)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using cpu")
        device = "cpu"

    if args.manifest is not None:
        print(f"Loading items from manifest (enforcing PA-only): {args.manifest}")
        items = load_items_from_manifest(args.data_root, args.manifest)
    else:
        print("Loading items from folders: data/raw/{healthy,cardiomegaly}")
        items = load_items_from_folders(args.data_root)

    # Smoke test subset (balanced)
    if args.smoke_test:
        rng = np.random.default_rng(args.seed)
        idx0 = [i for i, (_, y) in enumerate(items) if y == 0]
        idx1 = [i for i, (_, y) in enumerate(items) if y == 1]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        keep = idx0[:32] + idx1[:32]
        items = [items[i] for i in keep]

    y = np.array([yy for _, yy in items], dtype=np.int64)
    split = stratified_split(y, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)

    train_items = [items[i] for i in split.train]
    val_items = [items[i] for i in split.val]
    test_items = [items[i] for i in split.test]

    print("Split sizes")
    print(f"- train: {len(train_items)}")
    print(f"- val:   {len(val_items)}")
    print(f"- test:  {len(test_items)}")

    train_ds = DatasetA(train_items, image_size=args.image_size, train=True)
    val_ds = DatasetA(val_items, image_size=args.image_size, train=False)
    test_ds = DatasetA(test_items, image_size=args.image_size, train=False)

    pin = device.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = build_model(args.model).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    total_steps = max(1, len(train_loader) * int(args.epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    # Save best by AUROC if available; otherwise F1.
    best_key = "auc" if (SK is not None) else "f1"
    best_val = -float("inf")
    best_epoch = -1
    no_improve = 0

    args.out_weights.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: List[float] = []

        for x, yy in train_loader:
            x = x.to(device)
            yy = yy.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", enabled=device.startswith("cuda")):
                logits = model(x)
                loss = loss_fn(logits, yy)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            sched.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else math.nan
        val_m = eval_loop(model, val_loader, device=device, loss_fn=loss_fn)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_m["loss"]),
            "val_acc": float(val_m["acc"]),
            "val_f1": float(val_m["f1"]),
            "val_auc": float(val_m["auc"]),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss {row['train_loss']:.4f} | "
            f"val_loss {row['val_loss']:.4f} | "
            f"val_acc {row['val_acc']:.4f} | "
            f"val_f1 {row['val_f1']:.4f} | "
            f"val_auc {row['val_auc']:.4f}"
        )

        current = float(val_m[best_key])
        if current > best_val + 1e-6:
            best_val = current
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), args.out_weights)
        else:
            no_improve += 1

        if args.early_stop > 0 and no_improve >= int(args.early_stop):
            print(f"Early stopping: no improvement for {no_improve} epochs")
            break

    # Final test on best
    if args.out_weights.exists():
        model.load_state_dict(torch.load(args.out_weights, map_location=device, weights_only=True))

    test_m = eval_loop(model, test_loader, device=device, loss_fn=loss_fn)

    report: Dict[str, object] = {
        "dataset": "Dataset A (PA-only)",
        "source": str(args.manifest) if args.manifest else "folders:data/raw/*",
        "model": args.model,
        "counts": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
        "best": {"metric": best_key, "value": float(best_val), "epoch": int(best_epoch)},
        "test": {
            "loss": float(test_m["loss"]),
            "acc": float(test_m["acc"]),
            "f1": float(test_m["f1"]),
            "auc": float(test_m["auc"]),
        },
        "history": history,
        "weights": str(args.out_weights.as_posix()),
        "notes": {
            "sklearn_available": bool(SK is not None),
            "device": device,
            "tip": "Install scikit-learn for AUROC selection/metrics" if SK is None else "",
        },
    }

    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nDone")
    print(f"- saved weights: {args.out_weights}")
    print(f"- saved report:  {args.out_report}")
    print(
        "Test | "
        f"acc {report['test']['acc']:.4f} | "
        f"f1 {report['test']['f1']:.4f} | "
        f"auc {report['test']['auc']:.4f}"
    )


if __name__ == "__main__":
    # Avoid OpenMP duplicate warnings on Windows setups.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
