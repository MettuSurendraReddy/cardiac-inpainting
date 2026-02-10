#!/usr/bin/env python
"""Calibrate classifier confidence with temperature scaling.

Why:
- Softmax scores from neural nets are often overconfident.
- Temperature scaling fits a single scalar T on a *validation* set to make
  probabilities more realistic while keeping the same predicted labels.

What it does:
- Loads Dataset A from data/raw/{healthy,cardiomegaly} (optionally guided by a manifest)
- Recreates the same deterministic stratified split as train_dataset_a_classifier.py
- Computes logits on the validation split
- Fits temperature T by minimizing negative log-likelihood (cross-entropy)
- Writes a JSON calibration report

Typical usage:
  python scripts/calibrate_classifier_temperature.py --device cuda

Notes:
- This does NOT retrain the classifier. It only learns one scalar T.
- Use a held-out validation set (NOT test) for calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import CardiomegalyClassifier


@dataclass(frozen=True)
class Split:
    train: List[int]
    val: List[int]
    test: List[int]


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_items_from_folders(data_root: Path) -> List[Tuple[Path, int]]:
    healthy_dir = data_root / "raw" / "healthy"
    cardio_dir = data_root / "raw" / "cardiomegaly"

    if not healthy_dir.exists() or not cardio_dir.exists():
        raise FileNotFoundError(
            f"Expected Dataset A folders at {healthy_dir} and {cardio_dir}. "
            "Run: python scripts/export_nih_dataset_a_to_data_raw.py"
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


class DatasetA(Dataset):
    def __init__(self, items: List[Tuple[Path, int]], indices: List[int], image_size: int):
        self.items = items
        self.indices = indices
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
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        path, y = self.items[idx]
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        x = self.tf(img)
        return x, torch.tensor(y, dtype=torch.long)


def ece_from_probs(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error for max-prob confidence."""
    conf, pred = probs.max(dim=1)
    correct = (pred == y).float()

    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)

    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            bin_conf = conf[in_bin].mean()
            bin_acc = correct[in_bin].mean()
            bin_frac = in_bin.float().mean()
            ece += (bin_conf - bin_acc).abs().item() * bin_frac.item()

    return float(ece)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor, device: str) -> float:
    """Fit temperature T on validation logits by minimizing NLL."""
    logits = logits.to(device)
    y = y.to(device)

    # Optimize log(T) for numerical stability and positivity.
    log_t = torch.zeros((), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        t = torch.exp(log_t)
        loss = F.cross_entropy(logits / t, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = torch.exp(log_t).detach().item()

    # Avoid degenerate values.
    t = max(1e-3, min(float(t), 100.0))
    return float(t)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate classifier with temperature scaling")

    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest CSV (e.g. data/processed/NIH_A.csv or data/processed/Dataset_A.csv).",
    )

    p.add_argument(
        "--weights",
        type=Path,
        default=Path("models") / "classifier" / "dataset_a_classifier.pt",
        help="Path to classifier weights",
    )

    p.add_argument("--image-size", type=int, default=224)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda")

    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / "classifier" / "dataset_a_calibration.json",
        help="Where to save calibration report JSON",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Classifier weights not found: {args.weights}")

    items = (
        load_items_from_manifest(args.data_root, args.manifest)
        if args.manifest is not None
        else load_items_from_folders(args.data_root)
    )

    y = np.array([label for _, label in items], dtype=np.int64)
    split = stratified_split(y, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed)

    if len(split.val) == 0:
        raise RuntimeError("Validation split is empty. Increase dataset size or val-frac.")

    ds_val = DatasetA(items=items, indices=split.val, image_size=args.image_size)
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
    )

    # Load classifier (temperature=1.0) and use raw logits.
    clf = CardiomegalyClassifier(weights_path=args.weights, device=args.device, image_size=args.image_size, temperature=1.0)
    clf.model.eval()

    all_logits: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            logits = clf.model(xb)
            all_logits.append(logits.detach().cpu())
            all_y.append(yb.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_t = torch.cat(all_y, dim=0)

    probs_1 = torch.softmax(logits, dim=1)
    nll_1 = F.cross_entropy(logits, y_t).item()
    ece_1 = ece_from_probs(probs_1, y_t)

    t = fit_temperature(logits, y_t, device=args.device)

    probs_t = torch.softmax(logits / t, dim=1)
    nll_t = F.cross_entropy(logits / t, y_t).item()
    ece_t = ece_from_probs(probs_t, y_t)

    out = {
        "dataset": "Dataset A (from data/raw)",
        "weights": str(args.weights).replace("\\\\", "/"),
        "seed": int(args.seed),
        "split": {"train_frac": float(args.train_frac), "val_frac": float(args.val_frac)},
        "n": {"total": int(len(items)), "val": int(len(split.val))},
        "temperature": float(t),
        "metrics": {
            "val_nll_before": float(nll_1),
            "val_nll_after": float(nll_t),
            "val_ece_before": float(ece_1),
            "val_ece_after": float(ece_t),
        },
        "notes": "Temperature scaling changes probabilities (confidence) but not argmax class predictions.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Calibration complete")
    print(f"- T = {t:.4g}")
    print(f"- NLL: {nll_1:.4f} -> {nll_t:.4f}")
    print(f"- ECE: {ece_1:.4f} -> {ece_t:.4f}")
    print(f"- Saved: {args.out}")


if __name__ == "__main__":
    main()
