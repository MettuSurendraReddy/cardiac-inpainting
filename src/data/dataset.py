"""
PyTorch Dataset classes for the Cardiac Inpainting project.

Provides datasets for training the inpainting model on healthy chest X-rays.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class CardiacInpaintingDataset(Dataset):
    """
    Dataset for training the inpainting model.
    
    For each sample, provides:
    - image: The original chest X-ray (healthy)
    - mask: Binary mask of the heart region
    - masked_image: Image with heart region masked out
    
    Training strategy: Train on HEALTHY images only.
    The model learns to generate healthy hearts by reconstructing
    masked-out heart regions in healthy X-rays.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        image_size: int = 512,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        return_paths: bool = False,
        dilate_mask_range: Optional[Tuple[float, float]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing chest X-ray images
            masks_dir: Directory containing corresponding heart masks
            image_size: Target size for images (square)
            transform: Optional transforms to apply to images
            mask_transform: Optional transforms to apply to masks
            return_paths: If True, include file paths in output
            dilate_mask_range: Tuple (min, max) for random dilation factor.
                               E.g., (0.3, 0.8) dilates mask by 30-80%.
                               This simulates cardiomegaly-sized masks during training,
                               teaching the model to generate smaller hearts + fill anatomy.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_paths = return_paths
        self.dilate_mask_range = dilate_mask_range
        
        # Find all image files
        self.image_paths = self._find_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")
    
    def _find_images(self) -> List[Path]:
        """Find all image files that have corresponding masks."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        images = []
        
        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() not in valid_extensions:
                continue
            
            # Check if corresponding mask exists
            mask_path = self._get_mask_path(img_path)
            if mask_path.exists():
                images.append(img_path)
        
        return sorted(images)
    
    def _get_mask_path(self, image_path: Path) -> Path:
        """Get the mask path corresponding to an image."""
        # Try same filename in masks directory
        mask_path = self.masks_dir / image_path.name
        if mask_path.exists():
            return mask_path
        
        # Try with different extension
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = self.masks_dir / f"{image_path.stem}{ext}"
            if mask_path.exists():
                return mask_path
        
        # Default: same name as image
        return self.masks_dir / image_path.name
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - image: Original image tensor [C, H, W]
            - mask: Binary mask tensor [1, H, W] (1 = heart region)
            - masked_image: Image with heart masked out [C, H, W]
            - path (optional): Original image path
        """
        image_path = self.image_paths[idx]
        mask_path = self._get_mask_path(image_path)
        
        # Load image and mask
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform is not None:
            # Apply same spatial transforms to both image and mask
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        
        # Dilate mask to simulate cardiomegaly-sized masks during training
        # This teaches the model to generate smaller hearts + fill surrounding anatomy
        if self.dilate_mask_range is not None:
            mask = self._dilate_mask(mask, self.dilate_mask_range)
        
        # Convert to tensors
        image = self._to_tensor(image)
        mask = self._to_tensor(mask, is_mask=True)
        
        # Create masked image (image with heart region zeroed out)
        masked_image = image * (1 - mask)
        
        result = {
            'image': image,
            'mask': mask,
            'masked_image': masked_image
        }
        
        if self.return_paths:
            result['path'] = str(image_path)
        
        return result
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess an image."""
        image = Image.open(path)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and normalize to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.0
        
        return image
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and preprocess a mask."""
        mask = Image.open(path)
        
        # Convert to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')
        
        # Resize using nearest neighbor to preserve binary values
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        
        # Convert to numpy array and binarize
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)
        
        return mask
    
    def _to_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        if array.ndim == 2:
            # Add channel dimension [H, W] -> [1, H, W]
            array = array[np.newaxis, ...]
        
        return torch.from_numpy(array)
    
    def _dilate_mask(self, mask: np.ndarray, dilate_range: Tuple[float, float]) -> np.ndarray:
        """
        Randomly dilate the mask to simulate cardiomegaly-sized masks.
        
        This is crucial for training: by dilating healthy heart masks to be larger
        (like cardiomegaly hearts), the model learns to:
        1. Generate a smaller heart than the mask
        2. Fill the remaining space with correct anatomy (lungs, ribs, vessels)
        
        Args:
            mask: Binary mask [H, W] with values 0 or 1
            dilate_range: (min_factor, max_factor) for dilation
                         E.g., (0.3, 0.8) means dilate by 30-80%
        
        Returns:
            Dilated mask
        """
        min_factor, max_factor = dilate_range
        dilation_factor = random.uniform(min_factor, max_factor)
        
        # Calculate kernel size based on mask size and dilation factor
        # Larger factor = more dilation
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find the approximate size of the heart
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Get bounding box of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        heart_size = max(w, h)
        
        # Calculate kernel size: dilate by factor of current heart size
        kernel_size = int(heart_size * dilation_factor)
        kernel_size = max(3, kernel_size)  # Minimum kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd
        
        # Create elliptical kernel (more natural for heart shape)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate the mask
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        return (dilated / 255.0).astype(np.float32)


class InferenceDataset(Dataset):
    """
    Dataset for inference on cardiomegaly images.
    
    Unlike the training dataset, this doesn't require pre-existing masks.
    Masks will be generated by the segmentation model during inference.
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path],
        image_size: int = 512,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the inference dataset.
        
        Args:
            images_dir: Directory containing chest X-ray images
            image_size: Target size for images (square)
            transform: Optional transforms to apply
        """
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Find all image files
        self.image_paths = self._find_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")
    
    def _find_images(self) -> List[Path]:
        """Find all image files."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        images = []
        
        for img_path in self.images_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                images.append(img_path)
        
        return sorted(images)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
            - image: Image tensor [C, H, W]
            - path: Original image path
            - original_size: Original image size (W, H)
        """
        image_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(image_path)
        original_size = image.size  # (W, H)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        
        # Convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        image = torch.from_numpy(image)
        
        return {
            'image': image,
            'path': str(image_path),
            'original_size': original_size
        }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader
