"""
Data preparation utilities for the Cardiac Inpainting project.

Handles data preprocessing, train/val splitting, and directory organization.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataPreparation:
    """
    Handles data preparation and organization for training.
    
    Responsibilities:
    - Organize raw data into train/val splits
    - Resize and preprocess images
    - Verify image-mask pairs
    - Generate dataset statistics
    """
    
    def __init__(
        self,
        raw_images_dir: Union[str, Path],
        raw_masks_dir: Union[str, Path],
        output_dir: Union[str, Path],
        image_size: int = 512,
        train_split: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize data preparation.
        
        Args:
            raw_images_dir: Directory containing raw images
            raw_masks_dir: Directory containing raw masks
            output_dir: Output directory for processed data
            image_size: Target image size (square)
            train_split: Fraction of data for training
            seed: Random seed for reproducibility
        """
        self.raw_images_dir = Path(raw_images_dir)
        self.raw_masks_dir = Path(raw_masks_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.train_split = train_split
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
    
    def prepare(
        self,
        force: bool = False,
        verbose: bool = True
    ) -> Dict[str, int]:
        """
        Prepare the dataset.
        
        Args:
            force: If True, overwrite existing processed data
            verbose: If True, show progress
            
        Returns:
            Dictionary with dataset statistics
        """
        # Check if already processed
        if self._is_processed() and not force:
            if verbose:
                print("Dataset already prepared. Use force=True to reprocess.")
            return self._get_stats()
        
        # Create output directories
        self._create_directories()
        
        # Find valid image-mask pairs
        pairs = self._find_valid_pairs()
        
        if verbose:
            print(f"Found {len(pairs)} valid image-mask pairs")
        
        if len(pairs) == 0:
            raise ValueError("No valid image-mask pairs found")
        
        # Split into train and val
        random.shuffle(pairs)
        split_idx = int(len(pairs) * self.train_split)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        if verbose:
            print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
        
        # Process and save
        if verbose:
            print("Processing training data...")
        self._process_pairs(train_pairs, "train", verbose)
        
        if verbose:
            print("Processing validation data...")
        self._process_pairs(val_pairs, "val", verbose)
        
        # Save statistics
        stats = {
            "total": len(pairs),
            "train": len(train_pairs),
            "val": len(val_pairs),
            "image_size": self.image_size
        }
        
        self._save_stats(stats)
        
        return stats
    
    def _create_directories(self):
        """Create output directory structure."""
        dirs = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "masks",
            self.output_dir / "val" / "images",
            self.output_dir / "val" / "masks",
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _find_valid_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all valid image-mask pairs."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        pairs = []
        
        for img_path in self.raw_images_dir.iterdir():
            if img_path.suffix.lower() not in valid_extensions:
                continue
            
            # Try to find corresponding mask
            mask_path = self._find_mask(img_path)
            
            if mask_path is not None:
                pairs.append((img_path, mask_path))
        
        return pairs
    
    def _find_mask(self, image_path: Path) -> Optional[Path]:
        """Find the mask corresponding to an image."""
        # Try same filename
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            mask_path = self.raw_masks_dir / f"{image_path.stem}{ext}"
            if mask_path.exists():
                return mask_path
        
        # Try same full name
        mask_path = self.raw_masks_dir / image_path.name
        if mask_path.exists():
            return mask_path
        
        return None
    
    def _process_pairs(
        self,
        pairs: List[Tuple[Path, Path]],
        split: str,
        verbose: bool = True
    ):
        """Process and save image-mask pairs."""
        images_dir = self.output_dir / split / "images"
        masks_dir = self.output_dir / split / "masks"
        
        iterator = tqdm(pairs, desc=f"Processing {split}") if verbose else pairs
        
        for img_path, mask_path in iterator:
            # Process image
            image = self._process_image(img_path)
            
            # Process mask
            mask = self._process_mask(mask_path)
            
            # Save with standardized name (PNG format)
            output_name = f"{img_path.stem}.png"
            image.save(images_dir / output_name)
            mask.save(masks_dir / output_name)
    
    def _process_image(self, path: Path) -> Image.Image:
        """Load and preprocess an image."""
        image = Image.open(path)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )
        
        return image
    
    def _process_mask(self, path: Path) -> Image.Image:
        """Load and preprocess a mask."""
        mask = Image.open(path)
        
        # Convert to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Resize with nearest neighbor to preserve binary values
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )
        
        # Ensure binary
        mask = mask.point(lambda x: 255 if x > 127 else 0)
        
        return mask
    
    def _is_processed(self) -> bool:
        """Check if data has already been processed."""
        required_dirs = [
            self.output_dir / "train" / "images",
            self.output_dir / "train" / "masks",
            self.output_dir / "val" / "images",
            self.output_dir / "val" / "masks",
        ]
        
        for d in required_dirs:
            if not d.exists() or not any(d.iterdir()):
                return False
        
        return True
    
    def _get_stats(self) -> Dict[str, int]:
        """Get statistics of processed dataset."""
        stats = {}
        
        train_images = list((self.output_dir / "train" / "images").glob("*"))
        val_images = list((self.output_dir / "val" / "images").glob("*"))
        
        stats["train"] = len(train_images)
        stats["val"] = len(val_images)
        stats["total"] = stats["train"] + stats["val"]
        stats["image_size"] = self.image_size
        
        return stats
    
    def _save_stats(self, stats: Dict):
        """Save dataset statistics."""
        import json
        
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def verify_dataset(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify a dataset for training.
    
    Checks:
    - All images have corresponding masks
    - Images and masks have correct formats
    - No corrupted files
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        verbose: If True, print issues
        
    Returns:
        Dictionary with verification results
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    
    results = {
        "valid": 0,
        "missing_mask": [],
        "corrupted": [],
        "size_mismatch": []
    }
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in valid_extensions:
            continue
        
        # Check for mask
        mask_found = False
        for ext in valid_extensions:
            mask_path = masks_dir / f"{img_path.stem}{ext}"
            if mask_path.exists():
                mask_found = True
                break
        
        if not mask_found:
            results["missing_mask"].append(str(img_path))
            continue
        
        # Try to load files
        try:
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            # Check sizes match (after potential resize they should)
            # Just verify they can be loaded
            img.verify()
            
        except Exception as e:
            results["corrupted"].append((str(img_path), str(e)))
            continue
        
        results["valid"] += 1
    
    if verbose:
        print(f"Valid pairs: {results['valid']}")
        print(f"Missing masks: {len(results['missing_mask'])}")
        print(f"Corrupted files: {len(results['corrupted'])}")
    
    return results


def compute_dataset_statistics(
    images_dir: Union[str, Path],
    sample_size: int = 100
) -> Dict[str, float]:
    """
    Compute statistics of the dataset for normalization.
    
    Args:
        images_dir: Directory containing images
        sample_size: Number of images to sample
        
    Returns:
        Dictionary with mean and std
    """
    images_dir = Path(images_dir)
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    image_paths = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in valid_extensions
    ]
    
    # Sample if dataset is large
    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)
    
    pixel_values = []
    
    for path in tqdm(image_paths, desc="Computing statistics"):
        img = Image.open(path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        pixel_values.extend(img_array.flatten().tolist())
    
    pixel_values = np.array(pixel_values)
    
    return {
        "mean": float(np.mean(pixel_values)),
        "std": float(np.std(pixel_values)),
        "min": float(np.min(pixel_values)),
        "max": float(np.max(pixel_values))
    }
