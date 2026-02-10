"""
Anatomical validation for generated chest X-rays.

Provides tools for validating that generated images are anatomically correct,
primarily through Cardiothoracic Ratio (CTR) calculation.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image


class AnatomicalValidator:
    """
    Validates generated images are anatomically plausible.
    
    Primary validation is through Cardiothoracic Ratio (CTR):
    - CTR < 0.5: Healthy heart
    - CTR ≥ 0.5: Cardiomegaly
    
    CTR = Heart Width / Chest Width (measured at widest points)
    """
    
    def __init__(
        self,
        segmenter=None,
        min_ctr: float = 0.35,
        max_ctr: float = 0.50
    ):
        """
        Initialize the validator.
        
        Args:
            segmenter: ChexMaskSegmenter instance for segmentation
            min_ctr: Minimum acceptable CTR (below = unrealistically small heart)
            max_ctr: Maximum acceptable CTR (above = still cardiomegaly)
        """
        self.segmenter = segmenter
        self.min_ctr = min_ctr
        self.max_ctr = max_ctr
    
    def calculate_ctr(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        heart_mask: Optional[np.ndarray] = None,
        lung_mask: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """
        Calculate Cardiothoracic Ratio.
        
        CTR = max_heart_width / max_chest_width
        
        Args:
            image: Input chest X-ray (or path to it)
            heart_mask: Pre-computed heart mask (optional)
            lung_mask: Pre-computed lung mask (optional)
            
        Returns:
            CTR value (0.0-1.0), or None if calculation fails
        """
        # Get masks if not provided
        if heart_mask is None or lung_mask is None:
            if self.segmenter is None:
                raise ValueError(
                    "Either provide masks or initialize with a segmenter"
                )
            masks = self.segmenter.segment(image)
            heart_mask = masks['heart']
            lung_mask = masks['lungs']
        
        # Calculate heart width
        heart_width = self._get_max_width(heart_mask)
        
        # Calculate chest width using lungs
        chest_width = self._get_max_width(lung_mask)
        
        if chest_width == 0:
            return None
        
        return heart_width / chest_width
    
    def _get_max_width(self, mask: np.ndarray) -> int:
        """
        Get the maximum horizontal width of a mask.
        
        Args:
            mask: Binary mask [H, W]
            
        Returns:
            Maximum width in pixels
        """
        # Find columns that contain the mask
        col_sums = mask.sum(axis=0)
        mask_cols = np.where(col_sums > 0)[0]
        
        if len(mask_cols) == 0:
            return 0
        
        # Width is the difference between rightmost and leftmost columns
        return mask_cols[-1] - mask_cols[0]
    
    def calculate_ctr_from_masks(
        self,
        heart_mask: np.ndarray,
        lung_mask: np.ndarray
    ) -> Optional[float]:
        """
        Calculate CTR directly from masks.
        
        Args:
            heart_mask: Binary heart mask [H, W]
            lung_mask: Binary combined lung mask [H, W]
            
        Returns:
            CTR value or None
        """
        heart_width = self._get_max_width(heart_mask)
        chest_width = self._get_max_width(lung_mask)
        
        if chest_width == 0:
            return None
        
        return heart_width / chest_width
    
    def validate(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        heart_mask: Optional[np.ndarray] = None,
        lung_mask: Optional[np.ndarray] = None,
        min_ctr: Optional[float] = None,
        max_ctr: Optional[float] = None
    ) -> Tuple[bool, Dict]:
        """
        Check if image passes anatomical validation.
        
        Args:
            image: Generated image
            heart_mask: Pre-computed heart mask (optional)
            lung_mask: Pre-computed lung mask (optional)
            min_ctr: Override minimum acceptable CTR
            max_ctr: Override maximum acceptable CTR
            
        Returns:
            Tuple of (is_valid, details_dict)
        """
        min_ctr = min_ctr if min_ctr is not None else self.min_ctr
        max_ctr = max_ctr if max_ctr is not None else self.max_ctr
        
        # Calculate CTR
        ctr = self.calculate_ctr(image, heart_mask, lung_mask)
        
        if ctr is None:
            return False, {
                'valid': False,
                'error': 'Could not calculate CTR',
                'ctr': None
            }
        
        # Check bounds
        if ctr < min_ctr:
            return False, {
                'valid': False,
                'error': f'Heart too small (CTR={ctr:.3f} < {min_ctr})',
                'ctr': ctr
            }
        
        if ctr > max_ctr:
            return False, {
                'valid': False,
                'error': f'Still cardiomegaly (CTR={ctr:.3f} > {max_ctr})',
                'ctr': ctr
            }
        
        return True, {
            'valid': True,
            'status': 'healthy',
            'ctr': ctr
        }
    
    def get_heart_bounds(
        self,
        heart_mask: np.ndarray
    ) -> Dict[str, int]:
        """
        Get bounding box of heart region.
        
        Args:
            heart_mask: Binary heart mask [H, W]
            
        Returns:
            Dictionary with 'left', 'right', 'top', 'bottom', 'width', 'height'
        """
        rows = np.where(heart_mask.sum(axis=1) > 0)[0]
        cols = np.where(heart_mask.sum(axis=0) > 0)[0]
        
        if len(rows) == 0 or len(cols) == 0:
            return {
                'left': 0, 'right': 0, 'top': 0, 'bottom': 0,
                'width': 0, 'height': 0
            }
        
        return {
            'left': int(cols[0]),
            'right': int(cols[-1]),
            'top': int(rows[0]),
            'bottom': int(rows[-1]),
            'width': int(cols[-1] - cols[0]),
            'height': int(rows[-1] - rows[0])
        }
    
    def get_detailed_measurements(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        heart_mask: Optional[np.ndarray] = None,
        lung_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Get detailed anatomical measurements.
        
        Args:
            image: Input chest X-ray
            heart_mask: Pre-computed heart mask (optional)
            lung_mask: Pre-computed lung mask (optional)
            
        Returns:
            Dictionary with detailed measurements
        """
        # Get masks if not provided
        if heart_mask is None or lung_mask is None:
            if self.segmenter is None:
                raise ValueError("Segmenter required for detailed measurements")
            masks = self.segmenter.segment(image)
            heart_mask = masks['heart']
            lung_mask = masks['lungs']
        
        # Get image dimensions
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img_height, img_width = image.shape[:2]
        
        # Calculate measurements
        heart_width = self._get_max_width(heart_mask)
        chest_width = self._get_max_width(lung_mask)
        heart_bounds = self.get_heart_bounds(heart_mask)
        
        # Heart area (number of pixels)
        heart_area = heart_mask.sum()
        total_area = img_width * img_height
        heart_ratio = heart_area / total_area if total_area > 0 else 0
        
        # CTR
        ctr = heart_width / chest_width if chest_width > 0 else None
        
        return {
            'image_size': {'width': img_width, 'height': img_height},
            'heart': {
                'width': heart_width,
                'height': heart_bounds['height'],
                'area_pixels': int(heart_area),
                'area_ratio': float(heart_ratio),
                'bounds': heart_bounds
            },
            'chest': {
                'width': chest_width
            },
            'ctr': ctr,
            'is_healthy': ctr is not None and ctr < 0.5,
            'is_cardiomegaly': ctr is not None and ctr >= 0.5
        }


def calculate_ctr_simple(
    heart_mask: np.ndarray,
    lung_mask: np.ndarray
) -> Optional[float]:
    """
    Simple CTR calculation function (no class needed).
    
    Args:
        heart_mask: Binary heart mask [H, W]
        lung_mask: Binary combined lung mask [H, W]
        
    Returns:
        CTR value or None
    """
    # Find heart width
    heart_cols = np.where(heart_mask.sum(axis=0) > 0)[0]
    heart_width = heart_cols[-1] - heart_cols[0] if len(heart_cols) > 0 else 0
    
    # Find chest width
    lung_cols = np.where(lung_mask.sum(axis=0) > 0)[0]
    chest_width = lung_cols[-1] - lung_cols[0] if len(lung_cols) > 0 else 0
    
    if chest_width == 0:
        return None
    
    return heart_width / chest_width


def is_healthy_ctr(ctr: float, threshold: float = 0.5) -> bool:
    """
    Check if a CTR value indicates a healthy heart.
    
    Args:
        ctr: Cardiothoracic ratio
        threshold: Maximum healthy CTR (default 0.5)
        
    Returns:
        True if healthy, False if cardiomegaly
    """
    return ctr < threshold


def ctr_to_diagnosis(ctr: float) -> str:
    """
    Convert CTR to a diagnostic string.
    
    Args:
        ctr: Cardiothoracic ratio
        
    Returns:
        Diagnostic string
    """
    if ctr < 0.35:
        return "abnormally_small"
    elif ctr < 0.45:
        return "normal_small"
    elif ctr < 0.50:
        return "normal_borderline"
    elif ctr < 0.55:
        return "mild_cardiomegaly"
    elif ctr < 0.60:
        return "moderate_cardiomegaly"
    else:
        return "severe_cardiomegaly"
