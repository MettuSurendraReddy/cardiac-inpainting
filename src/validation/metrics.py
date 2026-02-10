"""
Evaluation metrics for the Cardiac Inpainting project.

Provides metrics for evaluating the quality of generated images.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class EvaluationMetrics:
    """
    Collection of metrics for evaluating inpainting quality.
    
    Tracks:
    - Success rate (valid outputs / total inputs)
    - CTR distribution (mean, std)
    - Classifier accuracy
    - Visual quality metrics (SSIM on non-masked regions)
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.results = []
        self.ctrs = []
        self.classifier_predictions = []
        self.ssim_scores = []
        self.success_count = 0
        self.total_count = 0
    
    def add_result(
        self,
        success: bool,
        ctr: Optional[float] = None,
        classifier_pred: Optional[str] = None,
        classifier_conf: Optional[float] = None,
        ssim: Optional[float] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a single result.
        
        Args:
            success: Whether generation was successful
            ctr: Cardiothoracic ratio of generated image
            classifier_pred: Classifier prediction ('healthy' or 'cardiomegaly')
            classifier_conf: Classifier confidence
            ssim: SSIM score on non-masked regions
            metadata: Additional metadata
        """
        self.total_count += 1
        
        if success:
            self.success_count += 1
            
            if ctr is not None:
                self.ctrs.append(ctr)
            
            if classifier_pred is not None:
                self.classifier_predictions.append({
                    'prediction': classifier_pred,
                    'confidence': classifier_conf
                })
            
            if ssim is not None:
                self.ssim_scores.append(ssim)
        
        self.results.append({
            'success': success,
            'ctr': ctr,
            'classifier_pred': classifier_pred,
            'classifier_conf': classifier_conf,
            'ssim': ssim,
            'metadata': metadata
        })
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {
            'total': self.total_count,
            'successful': self.success_count,
            'failed': self.total_count - self.success_count,
            'success_rate': self.success_count / max(1, self.total_count)
        }
        
        # CTR statistics
        if self.ctrs:
            summary['ctr'] = {
                'mean': float(np.mean(self.ctrs)),
                'std': float(np.std(self.ctrs)),
                'min': float(np.min(self.ctrs)),
                'max': float(np.max(self.ctrs)),
                'median': float(np.median(self.ctrs))
            }
        else:
            summary['ctr'] = None
        
        # Classifier statistics
        if self.classifier_predictions:
            healthy_count = sum(
                1 for p in self.classifier_predictions
                if p['prediction'] == 'healthy'
            )
            summary['classifier'] = {
                'healthy_count': healthy_count,
                'healthy_rate': healthy_count / len(self.classifier_predictions),
                'avg_confidence': float(np.mean([
                    p['confidence'] for p in self.classifier_predictions
                    if p['confidence'] is not None
                ]))
            }
        else:
            summary['classifier'] = None
        
        # SSIM statistics
        if self.ssim_scores:
            summary['ssim'] = {
                'mean': float(np.mean(self.ssim_scores)),
                'std': float(np.std(self.ssim_scores)),
                'min': float(np.min(self.ssim_scores)),
                'max': float(np.max(self.ssim_scores))
            }
        else:
            summary['ssim'] = None
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        print(f"\nOverall:")
        print(f"  Total samples: {summary['total']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        if summary['ctr']:
            print(f"\nCTR Distribution:")
            print(f"  Mean: {summary['ctr']['mean']:.3f}")
            print(f"  Std: {summary['ctr']['std']:.3f}")
            print(f"  Range: [{summary['ctr']['min']:.3f}, {summary['ctr']['max']:.3f}]")
        
        if summary['classifier']:
            print(f"\nClassifier Results:")
            print(f"  Healthy rate: {summary['classifier']['healthy_rate']:.1%}")
            print(f"  Avg confidence: {summary['classifier']['avg_confidence']:.3f}")
        
        if summary['ssim']:
            print(f"\nSSIM (non-masked regions):")
            print(f"  Mean: {summary['ssim']['mean']:.4f}")
            print(f"  Std: {summary['ssim']['std']:.4f}")
        
        print("=" * 50 + "\n")


def calculate_ssim(
    image1: Union[np.ndarray, Image.Image],
    image2: Union[np.ndarray, Image.Image],
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0
) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: First image
        image2: Second image
        mask: Optional mask (1 = include in calculation, 0 = exclude)
        data_range: Data range of the images
        
    Returns:
        SSIM value (0-1, higher is more similar)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image required for SSIM. Install with: pip install scikit-image")
    
    # Convert to numpy arrays
    if isinstance(image1, Image.Image):
        image1 = np.array(image1.convert('L')).astype(np.float32) / 255.0
    if isinstance(image2, Image.Image):
        image2 = np.array(image2.convert('L')).astype(np.float32) / 255.0
    
    # Ensure same shape
    if image1.shape != image2.shape:
        from PIL import Image as PILImage
        img2_pil = PILImage.fromarray((image2 * 255).astype(np.uint8))
        img2_pil = img2_pil.resize((image1.shape[1], image1.shape[0]))
        image2 = np.array(img2_pil).astype(np.float32) / 255.0
    
    # Apply mask if provided
    if mask is not None:
        # Only compute on masked region
        mask_bool = mask > 0.5
        if not mask_bool.any():
            return 1.0  # No region to compare
        
        # Compute SSIM only on masked region
        return ssim(
            image1,
            image2,
            data_range=data_range,
            full=False
        )
    
    return ssim(image1, image2, data_range=data_range)


def calculate_ssim_masked(
    original: Union[np.ndarray, Image.Image],
    generated: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    compute_outside_mask: bool = True
) -> Dict[str, float]:
    """
    Calculate SSIM inside and outside the mask region.
    
    Args:
        original: Original image
        generated: Generated image
        mask: Binary mask (1 = inpainted region, 0 = preserved region)
        compute_outside_mask: If True, compute SSIM on preserved regions
        
    Returns:
        Dictionary with 'inside' and 'outside' SSIM values
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError("scikit-image required")
    
    # Convert to numpy
    if isinstance(original, Image.Image):
        original = np.array(original.convert('L')).astype(np.float32) / 255.0
    if isinstance(generated, Image.Image):
        generated = np.array(generated.convert('L')).astype(np.float32) / 255.0
    
    result = {}
    
    # SSIM outside mask (preserved region) - should be high
    if compute_outside_mask:
        outside_mask = mask < 0.5
        if outside_mask.any():
            # Set masked region to same value in both images
            orig_masked = original.copy()
            gen_masked = generated.copy()
            orig_masked[~outside_mask] = 0
            gen_masked[~outside_mask] = 0
            
            result['outside'] = ssim(orig_masked, gen_masked, data_range=1.0)
        else:
            result['outside'] = 1.0
    
    return result


def calculate_psnr(
    image1: Union[np.ndarray, Image.Image],
    image2: Union[np.ndarray, Image.Image],
    data_range: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        image1: First image
        image2: Second image
        data_range: Data range of the images
        
    Returns:
        PSNR value in dB (higher is better)
    """
    # Convert to numpy
    if isinstance(image1, Image.Image):
        image1 = np.array(image1.convert('L')).astype(np.float32) / 255.0
    if isinstance(image2, Image.Image):
        image2 = np.array(image2.convert('L')).astype(np.float32) / 255.0
    
    mse = np.mean((image1 - image2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    return 10 * np.log10((data_range ** 2) / mse)


def calculate_mae(
    image1: Union[np.ndarray, Image.Image],
    image2: Union[np.ndarray, Image.Image]
) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        MAE value
    """
    if isinstance(image1, Image.Image):
        image1 = np.array(image1.convert('L')).astype(np.float32) / 255.0
    if isinstance(image2, Image.Image):
        image2 = np.array(image2.convert('L')).astype(np.float32) / 255.0
    
    return np.mean(np.abs(image1 - image2))


def calculate_histogram_similarity(
    image1: Union[np.ndarray, Image.Image],
    image2: Union[np.ndarray, Image.Image],
    bins: int = 256
) -> float:
    """
    Calculate histogram similarity using correlation.
    
    Args:
        image1: First image
        image2: Second image
        bins: Number of histogram bins
        
    Returns:
        Correlation coefficient (0-1, higher is more similar)
    """
    import cv2
    
    # Convert to numpy
    if isinstance(image1, Image.Image):
        image1 = np.array(image1.convert('L'))
    if isinstance(image2, Image.Image):
        image2 = np.array(image2.convert('L'))
    
    # Ensure uint8
    if image1.dtype == np.float32 or image1.dtype == np.float64:
        image1 = (image1 * 255).astype(np.uint8)
    if image2.dtype == np.float32 or image2.dtype == np.float64:
        image2 = (image2 * 255).astype(np.uint8)
    
    # Calculate histograms
    hist1 = cv2.calcHist([image1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [bins], [0, 256])
    
    # Normalize
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Compare using correlation
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
