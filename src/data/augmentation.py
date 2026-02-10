"""
Data augmentation utilities for medical imaging.

Provides augmentation pipelines suitable for chest X-ray images,
carefully avoiding transforms that would create anatomically incorrect images.
"""

from typing import Dict, Optional, Tuple
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_train_augmentations(
    image_size: int = 512,
    rotation_limit: int = 5,
    brightness_limit: float = 0.1,
    contrast_limit: float = 0.1,
    scale_limit: float = 0.1
) -> "A.Compose":
    """
    Get training augmentations for chest X-ray images.
    
    These augmentations are carefully chosen to maintain anatomical correctness:
    - Slight rotation (±5°): Simulates slight patient positioning differences
    - Brightness/contrast: Simulates different X-ray exposure settings
    - Slight scale: Simulates different patient sizes
    
    NOT USED (would create anatomically incorrect images):
    - Horizontal flip: Would flip heart position (heart is on left side)
    - Vertical flip: Would create impossible anatomy
    - Large rotations: Patients are always upright in chest X-rays
    - Heavy cropping: Would remove important anatomical structures
    
    Args:
        image_size: Target image size
        rotation_limit: Maximum rotation in degrees
        brightness_limit: Brightness adjustment range
        contrast_limit: Contrast adjustment range
        scale_limit: Scale adjustment range (0.1 = 90%-110%)
        
    Returns:
        Albumentations Compose object
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for augmentations. "
            "Install with: pip install albumentations"
        )
    
    return A.Compose([
        # Geometric transforms (applied to both image and mask)
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=0,
            p=0.5
        ),
        
        # Intensity transforms (applied to image only)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
        ], p=0.5),
        
        # Slight blur (simulates different image quality)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.2),
        
        # Noise (simulates X-ray noise)
        A.GaussNoise(var_limit=(5, 20), p=0.3),
        
    ], additional_targets={'mask': 'mask'})


def get_val_augmentations(image_size: int = 512) -> "A.Compose":
    """
    Get validation/inference augmentations (no augmentation, just preprocessing).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required. Install with: pip install albumentations"
        )
    
    # No augmentations for validation - just return as-is
    return A.Compose([
        # No transforms - images are already preprocessed
    ], additional_targets={'mask': 'mask'})


def get_inference_augmentations(image_size: int = 512) -> "A.Compose":
    """
    Get augmentations for inference (same as validation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations Compose object
    """
    return get_val_augmentations(image_size)


class MedicalImageAugmenter:
    """
    Class-based augmenter for medical images with additional controls.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        mode: str = "train",
        **kwargs
    ):
        """
        Initialize the augmenter.
        
        Args:
            image_size: Target image size
            mode: "train", "val", or "inference"
            **kwargs: Additional arguments for augmentations
        """
        self.image_size = image_size
        self.mode = mode
        
        if mode == "train":
            self.transform = get_train_augmentations(image_size, **kwargs)
        else:
            self.transform = get_val_augmentations(image_size)
    
    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentations.
        
        Args:
            image: Input image (H, W) or (H, W, C)
            mask: Optional mask (H, W)
            
        Returns:
            Dictionary with 'image' and optionally 'mask'
        """
        if mask is not None:
            return self.transform(image=image, mask=mask)
        else:
            return self.transform(image=image)


# Simple numpy-based augmentations (fallback if albumentations not available)

def apply_brightness_contrast(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Apply brightness and contrast adjustment.
    
    Args:
        image: Input image (normalized to [0, 1])
        brightness: Brightness offset (-0.5 to 0.5)
        contrast: Contrast multiplier (0.5 to 1.5)
        
    Returns:
        Adjusted image
    """
    result = image * contrast + brightness
    return np.clip(result, 0, 1)


def apply_rotation(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    angle: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply rotation to image and mask.
    
    Args:
        image: Input image
        mask: Optional mask
        angle: Rotation angle in degrees
        
    Returns:
        Tuple of (rotated_image, rotated_mask)
    """
    from scipy.ndimage import rotate
    
    rotated_image = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
    
    if mask is not None:
        rotated_mask = rotate(mask, angle, reshape=False, order=0, mode='constant', cval=0)
        return rotated_image, rotated_mask
    
    return rotated_image, None


def apply_gaussian_noise(
    image: np.ndarray,
    std: float = 0.02
) -> np.ndarray:
    """
    Add Gaussian noise to image.
    
    Args:
        image: Input image (normalized to [0, 1])
        std: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    result = image + noise
    return np.clip(result, 0, 1)


def random_augment(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    rotation_limit: float = 5.0,
    brightness_limit: float = 0.1,
    contrast_range: Tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.02,
    augment_prob: float = 0.5
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply random augmentations without albumentations.
    
    This is a fallback for when albumentations is not available.
    
    Args:
        image: Input image (H, W), normalized to [0, 1]
        mask: Optional mask (H, W)
        rotation_limit: Maximum rotation in degrees
        brightness_limit: Brightness adjustment range
        contrast_range: (min_contrast, max_contrast)
        noise_std: Gaussian noise standard deviation
        augment_prob: Probability of applying each augmentation
        
    Returns:
        Tuple of (augmented_image, augmented_mask)
    """
    result_image = image.copy()
    result_mask = mask.copy() if mask is not None else None
    
    # Random rotation
    if np.random.random() < augment_prob:
        angle = np.random.uniform(-rotation_limit, rotation_limit)
        result_image, result_mask = apply_rotation(result_image, result_mask, angle)
    
    # Random brightness/contrast
    if np.random.random() < augment_prob:
        brightness = np.random.uniform(-brightness_limit, brightness_limit)
        contrast = np.random.uniform(*contrast_range)
        result_image = apply_brightness_contrast(result_image, brightness, contrast)
    
    # Random noise
    if np.random.random() < augment_prob:
        result_image = apply_gaussian_noise(result_image, noise_std)
    
    return result_image, result_mask
