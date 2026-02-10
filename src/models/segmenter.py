"""
CheXMask Segmenter wrapper.

Wraps the HybridGNet model for cardiac and lung segmentation from chest X-rays.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import torch
import scipy.sparse as sp


class ChexMaskSegmenter:
    """
    Wrapper for CheXMask HybridGNet.
    
    Provides:
    - Heart segmentation
    - Lung segmentation (for CTR calculation)
    - Mask preprocessing for inpainting
    """
    
    # Constants for the model
    RLUNG_POINTS = 44
    LLUNG_POINTS = 50
    HEART_POINTS = 26
    INPUT_SIZE = 1024
    
    def __init__(
        self,
        weights_dir: Union[str, Path],
        device: str = "cuda"
    ):
        """
        Initialize the segmenter.
        
        Args:
            weights_dir: Directory containing CheXMask weights
            device: Device to run inference on
        """
        self.device = device
        self.weights_dir = Path(weights_dir)
        
        # Add HybridGNet to path
        self._setup_paths()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
    
    def _setup_paths(self):
        """Set up import paths for HybridGNet."""
        # weights_dir = models/CheXmask-Database/Weights
        # chexmask_root = models/CheXmask-Database (parent of Weights)
        chexmask_root = self.weights_dir.parent
        hybridgnet_path = chexmask_root / "HybridGNet"
        
        if hybridgnet_path.exists():
            if str(hybridgnet_path) not in sys.path:
                sys.path.insert(0, str(hybridgnet_path))
            # Also add CheXmask-Database root for HybridGNet package imports
            if str(chexmask_root) not in sys.path:
                sys.path.insert(0, str(chexmask_root))
    
    def _load_model(self) -> torch.nn.Module:
        """Load the HybridGNet model."""
        # Import HybridGNet components
        try:
            from HybridGNet.models.HybridGNet2IGSC import Hybrid
            from HybridGNet.utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
        except ImportError:
            raise ImportError(
                "Could not import HybridGNet. Make sure the CheXmask-Database "
                "is properly set up in the models directory."
            )
        
        # Generate matrices
        A, AD, D, U = genMatrixesLungsHeart()
        N1 = A.shape[0]
        N2 = AD.shape[0]
        
        # Convert to sparse tensors
        A = sp.csc_matrix(A).tocoo()
        AD = sp.csc_matrix(AD).tocoo()
        D = sp.csc_matrix(D).tocoo()
        U = sp.csc_matrix(U).tocoo()
        
        D_ = [D.copy()]
        U_ = [U.copy()]
        
        # Model configuration
        config = {}
        config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
        A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
        
        A_t, D_t, U_t = (
            [scipy_to_torch_sparse(x).to(self.device) for x in X]
            for X in (A_, D_, U_)
        )
        
        config['latents'] = 64
        config['inputsize'] = self.INPUT_SIZE
        
        f = 32
        config['filters'] = [2, f, f, f, f//2, f//2, f//2]
        config['skip_features'] = f
        
        # Create model
        model = Hybrid(config.copy(), D_t, U_t, A_t).to(self.device)
        
        # Load weights
        weights_path = self.weights_dir / "SegmentationModel" / "bestMSE.pt"
        if weights_path.exists():
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        
        return model
    
    @torch.no_grad()
    def segment(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
        return_landmarks: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Segment an image into anatomical regions.
        
        Args:
            image: Input chest X-ray (PIL Image, numpy array, tensor, or path)
            return_landmarks: If True, also return landmark points
            
        Returns:
            Dictionary containing:
            - 'heart': Binary mask of heart [H, W]
            - 'left_lung': Binary mask of left lung [H, W]
            - 'right_lung': Binary mask of right lung [H, W]
            - 'lungs': Combined lung mask [H, W]
            - 'landmarks' (optional): Dictionary of landmark points
        """
        # Load and preprocess image
        img_array, original_size = self._preprocess_image(image)
        
        # Create input tensor
        input_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device).float()
        
        # Get landmarks from model
        output = self.model(input_tensor)
        if isinstance(output, (list, tuple)) and len(output) > 1:
            output = output[0]
        
        # Convert to numpy and scale to image coordinates
        landmarks = output.cpu().numpy().reshape(-1, 2) * self.INPUT_SIZE
        landmarks = landmarks.round().astype(np.int32)
        
        # Split landmarks into organs
        rl_landmarks = landmarks[:self.RLUNG_POINTS]
        ll_landmarks = landmarks[self.RLUNG_POINTS:self.RLUNG_POINTS + self.LLUNG_POINTS]
        h_landmarks = landmarks[self.RLUNG_POINTS + self.LLUNG_POINTS:]
        
        # Create masks
        heart_mask = self._landmarks_to_mask(h_landmarks, self.INPUT_SIZE)
        right_lung_mask = self._landmarks_to_mask(rl_landmarks, self.INPUT_SIZE)
        left_lung_mask = self._landmarks_to_mask(ll_landmarks, self.INPUT_SIZE)
        lungs_mask = np.logical_or(right_lung_mask, left_lung_mask).astype(np.uint8)
        
        # Resize masks to original size if needed
        if original_size != (self.INPUT_SIZE, self.INPUT_SIZE):
            heart_mask = self._resize_mask(heart_mask, original_size)
            right_lung_mask = self._resize_mask(right_lung_mask, original_size)
            left_lung_mask = self._resize_mask(left_lung_mask, original_size)
            lungs_mask = self._resize_mask(lungs_mask, original_size)
        
        result = {
            'heart': heart_mask,
            'left_lung': left_lung_mask,
            'right_lung': right_lung_mask,
            'lungs': lungs_mask
        }
        
        if return_landmarks:
            result['landmarks'] = {
                'heart': h_landmarks,
                'right_lung': rl_landmarks,
                'left_lung': ll_landmarks
            }
        
        return result
    
    def _preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess an image for segmentation.
        
        Returns:
            Tuple of (preprocessed_array, original_size)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            original_size = image.size  # (W, H)
            if image.mode != 'L':
                image = image.convert('L')
            img_array = np.array(image, dtype=np.float32)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.squeeze(0)
            img_array = image.cpu().numpy()
            if img_array.dtype != np.float32:
                img_array = img_array.astype(np.float32)
            original_size = (img_array.shape[1], img_array.shape[0])  # (W, H)
        else:
            img_array = image.astype(np.float32)
            original_size = (img_array.shape[1], img_array.shape[0])
        
        # Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Resize to model input size
        if img_array.shape != (self.INPUT_SIZE, self.INPUT_SIZE):
            img_array = cv2.resize(
                img_array,
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR
            )
        
        return img_array, original_size
    
    def _landmarks_to_mask(
        self,
        landmarks: np.ndarray,
        size: int
    ) -> np.ndarray:
        """
        Convert landmark points to a binary mask.
        
        Args:
            landmarks: Array of [N, 2] landmark points
            size: Size of the output mask
            
        Returns:
            Binary mask [size, size]
        """
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # Reshape for cv2.drawContours
        contour = landmarks.reshape(-1, 1, 2).astype(np.int32)
        
        # Fill the contour
        cv2.drawContours(mask, [contour], -1, 1, -1)
        
        return mask
    
    def _resize_mask(
        self,
        mask: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize a mask to target size.
        
        Args:
            mask: Binary mask
            target_size: (width, height)
            
        Returns:
            Resized mask
        """
        resized = cv2.resize(
            mask,
            target_size,
            interpolation=cv2.INTER_NEAREST
        )
        return (resized > 0.5).astype(np.uint8)
    
    def prepare_inpainting_mask(
        self,
        heart_mask: np.ndarray,
        dilation_factor: float = 0.1,
        feather_radius: int = 5
    ) -> np.ndarray:
        """
        Prepare heart mask for inpainting.
        
        Args:
            heart_mask: Binary heart mask [H, W]
            dilation_factor: Dilation relative to mask size
            feather_radius: Radius for edge feathering
            
        Returns:
            Processed mask for inpainting [H, W], float32, range [0, 1]
        """
        # Calculate dilation kernel size based on mask size
        mask_size = max(heart_mask.shape)
        kernel_size = max(3, int(mask_size * dilation_factor))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Dilate mask for safety margin
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(heart_mask.astype(np.uint8), kernel)
        
        # Feather edges for smooth blending
        if feather_radius > 0:
            # Convert to float and blur edges
            mask_float = dilated.astype(np.float32)
            
            # Apply Gaussian blur for feathering
            blur_size = feather_radius * 2 + 1
            feathered = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
            
            # Normalize to [0, 1]
            if feathered.max() > 0:
                feathered = feathered / feathered.max()
            
            return feathered
        
        return dilated.astype(np.float32)
    
    def get_combined_mask(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
        include_lungs: bool = True
    ) -> np.ndarray:
        """
        Get a combined segmentation mask.
        
        Args:
            image: Input chest X-ray
            include_lungs: If True, include lungs in the mask
            
        Returns:
            Combined mask with:
            - 0: Background
            - 1: Lungs (if include_lungs)
            - 2: Heart
        """
        masks = self.segment(image)
        
        combined = np.zeros_like(masks['heart'], dtype=np.uint8)
        
        if include_lungs:
            combined[masks['lungs'] > 0] = 1
        
        combined[masks['heart'] > 0] = 2
        
        return combined
