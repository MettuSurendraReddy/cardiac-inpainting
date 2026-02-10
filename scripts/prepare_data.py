#!/usr/bin/env python
"""
Data preparation script.

Prepares raw data for training by:
- Organizing into train/val splits
- Resizing images to consistent size
- Verifying image-mask pairs
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_path
from src.data.preparation import DataPreparation, verify_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Override raw images directory"
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        help="Override raw masks directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Training data split ratio"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Target image size"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if data exists"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify data, don't process"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    project_root = Path(__file__).parent.parent
    
    # Resolve paths
    images_dir = args.images_dir or resolve_path(
        config.paths.get('raw_data_dir', 'data/raw') + '/healthy',
        project_root
    )
    masks_dir = args.masks_dir or resolve_path(
        config.paths.get('masks_dir', 'data/masks') + '/healthy',
        project_root
    )
    output_dir = args.output_dir or resolve_path(
        config.paths.get('processed_data_dir', 'data/processed'),
        project_root
    )
    
    print("Data Preparation")
    print("=" * 50)
    print(f"Images directory: {images_dir}")
    print(f"Masks directory: {masks_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size}")
    print(f"Train split: {args.train_split}")
    print("=" * 50)
    
    # Verify only mode
    if args.verify_only:
        print("\nVerifying dataset...")
        results = verify_dataset(images_dir, masks_dir, verbose=True)
        
        if results['valid'] > 0:
            print(f"\n✓ Dataset is valid with {results['valid']} image-mask pairs")
        else:
            print("\n✗ No valid image-mask pairs found")
            sys.exit(1)
        return
    
    # Check if directories exist
    if not Path(images_dir).exists():
        print(f"\nError: Images directory not found: {images_dir}")
        print("Please ensure your healthy X-ray images are in this directory.")
        sys.exit(1)
    
    if not Path(masks_dir).exists():
        print(f"\nError: Masks directory not found: {masks_dir}")
        print("Please generate masks first using: python scripts/generate_masks.py")
        sys.exit(1)
    
    # Prepare data
    prep = DataPreparation(
        raw_images_dir=images_dir,
        raw_masks_dir=masks_dir,
        output_dir=output_dir,
        image_size=args.image_size,
        train_split=args.train_split,
        seed=config.get('seed', 42)
    )
    
    stats = prep.prepare(force=args.force, verbose=True)
    
    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"  Total samples: {stats['total']}")
    print(f"  Training samples: {stats['train']}")
    print(f"  Validation samples: {stats['val']}")
    print(f"  Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
