#!/usr/bin/env python
"""Export Dataset A -> data/raw/{healthy,cardiomegaly} (PA only).

Moved from sort_data/ into scripts/.

This replicates the logic from the notebook:
- Uses NIH/Kaggle metadata Data_Entry_2017.csv
- Filters to PA view only
- Keeps exact-label rows only: 'No Finding' vs 'Cardiomegaly'
- Enforces 1 image per patient (reduces leakage)
- Balances classes (same N healthy as cardiomegaly)
- Writes a manifest to data/processed/Dataset_A.csv
- Copies (or symlinks) images into:
  - data/raw/healthy
  - data/raw/cardiomegaly

Example:
  python scripts/export_nih_dataset_a_to_data_raw.py \
    --data-entry-csv "content/Data_Entry_2017.csv" \
    --images-root "content" \
    --out-root "data" \
    --max-per-class 500
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Dataset A to data/raw")
    p.add_argument(
        "--data-entry-csv",
        type=Path,
        default=Path("content") / "Data_Entry_2017.csv",
        help="Path to NIH metadata CSV (default: content/Data_Entry_2017.csv)",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        default=Path("content"),
        help=(
            "Root folder containing images_001..images_012 (each with an images/ subfolder) "
            "(default: content/)"
        ),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data"),
        help="Project data root (default: data)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for sampling/shuffling",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap per class (e.g. 500). Default: use all available cardiomegaly patients",
    )
    p.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy mode: copy or symlink (symlink may require admin on Windows)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="If set: do not copy/symlink files, only compute manifest and counts",
    )
    return p.parse_args()


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    raw_dir = out_root / "raw"
    paths = {
        "raw": raw_dir,
        "healthy": raw_dir / "healthy",
        "cardio": raw_dir / "cardiomegaly",
        "processed": out_root / "processed",
        "masks": out_root / "masks",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def build_dataset_a(df: pd.DataFrame, random_state: int, max_per_class: Optional[int]) -> pd.DataFrame:
    required = {"Image Index", "Finding Labels", "View Position", "Patient ID"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in Data_Entry_2017.csv: {sorted(missing)}")

    df_pa = df[df["View Position"] == "PA"].copy()

    labels = df_pa["Finding Labels"].fillna("")
    cardio = df_pa[labels == "Cardiomegaly"].copy()
    healthy = df_pa[labels == "No Finding"].copy()

    # 1 image per patient
    cardio = cardio.drop_duplicates(subset=["Patient ID"], keep="first")
    healthy = healthy.drop_duplicates(subset=["Patient ID"], keep="first")

    if len(cardio) == 0:
        raise ValueError("Found 0 cardiomegaly patients after filters (PA + exact-label).")

    n = len(cardio)
    if max_per_class is not None:
        n = min(n, int(max_per_class))

    if len(cardio) > n:
        cardio = cardio.sample(n=n, random_state=random_state)

    if len(healthy) < n:
        raise ValueError(f"Not enough healthy patients after filters: {len(healthy)} < {n}")

    healthy_bal = healthy.sample(n=n, random_state=random_state)

    dataset_a = pd.concat(
        [
            cardio.assign(class_name="cardiomegaly", class_id=1),
            healthy_bal.assign(class_name="healthy", class_id=0),
        ],
        ignore_index=True,
    )

    dataset_a = dataset_a.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return dataset_a


def find_image_path(images_root: Path, image_name: str) -> Optional[Path]:
    # Fast path for standard NIH/Kaggle layout
    for i in range(1, 13):
        candidate = images_root / f"images_{i:03d}" / "images" / image_name
        if candidate.exists():
            return candidate

    # Fallback: recursive search (can be slow)
    matches = list(images_root.glob(f"**/{image_name}"))
    return matches[0] if matches else None


def export_images(
    dataset_a: pd.DataFrame,
    images_root: Path,
    healthy_dir: Path,
    cardio_dir: Path,
    copy_mode: str,
    dry_run: bool,
) -> tuple[int, int, list[str]]:
    rows = dataset_a[["Image Index", "class_name"]].to_records(index=False)

    missing: list[str] = []
    copied = 0
    skipped_existing = 0

    for image_name, class_name in rows:
        image_name = str(image_name)
        src = find_image_path(images_root, image_name)
        if src is None:
            missing.append(image_name)
            continue

        dst_dir = cardio_dir if class_name == "cardiomegaly" else healthy_dir
        dst = dst_dir / image_name

        if dst.exists():
            skipped_existing += 1
            continue

        if dry_run:
            copied += 1
            continue

        if copy_mode == "symlink":
            try:
                dst.symlink_to(src)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

        copied += 1

    return copied, skipped_existing, missing


def main() -> None:
    args = parse_args()

    if not args.data_entry_csv.exists():
        raise FileNotFoundError(f"Not found: {args.data_entry_csv.resolve()}")
    if not args.images_root.exists():
        raise FileNotFoundError(f"Not found: {args.images_root.resolve()}")

    paths = ensure_dirs(args.out_root)

    print("Export: Dataset A → data/raw/{healthy,cardiomegaly} (PA only)")
    print("=" * 60)
    print(f"Data entry CSV: {args.data_entry_csv}")
    print(f"Images root:    {args.images_root}")
    print(f"Out root:       {args.out_root}")
    print(f"Copy mode:      {args.copy_mode}")
    print(f"Dry run:        {args.dry_run}")
    print(f"Max/class:      {args.max_per_class}")

    df = pd.read_csv(args.data_entry_csv)
    dataset_a = build_dataset_a(df, random_state=args.random_state, max_per_class=args.max_per_class)

    out_csv = paths["processed"] / "Dataset_A.csv"
    dataset_a.to_csv(out_csv, index=False)

    print("\n✅ Manifest written")
    print(f" - {out_csv}")
    print(dataset_a["class_name"].value_counts())

    copied, skipped, missing = export_images(
        dataset_a=dataset_a,
        images_root=args.images_root,
        healthy_dir=paths["healthy"],
        cardio_dir=paths["cardio"],
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
    )

    print("\n✅ Export done")
    print(f" - Copied/linked: {copied}")
    print(f" - Skipped (already existed): {skipped}")
    print(f" - Missing source files: {len(missing)}")
    if missing:
        print("Examples missing:", missing[:10])


if __name__ == "__main__":
    main()
