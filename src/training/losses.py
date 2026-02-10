"""
Custom loss functions for training the inpainting model.

Provides losses specifically designed for medical image inpainting.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    MSE Loss computed only on masked (inpainted) regions.
    
    This ensures the model focuses on learning to generate
    the heart region correctly.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked MSE loss.
        
        Args:
            pred: Predicted values [B, C, H, W]
            target: Target values [B, C, H, W]
            mask: Binary mask [B, 1, H, W] where 1 = compute loss
            
        Returns:
            Loss value
        """
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        # Apply mask
        masked_error = squared_error * mask
        
        if self.reduction == 'none':
            return masked_error
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:  # mean
            # Mean over masked region only
            num_elements = mask.sum().clamp(min=1)
            return masked_error.sum() / num_elements


class MaskedL1Loss(nn.Module):
    """
    L1 Loss computed only on masked regions.
    
    L1 tends to produce sharper results than MSE.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize the loss.
        
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked L1 loss.
        
        Args:
            pred: Predicted values [B, C, H, W]
            target: Target values [B, C, H, W]
            mask: Binary mask [B, 1, H, W]
            
        Returns:
            Loss value
        """
        # Compute absolute error
        abs_error = torch.abs(pred - target)
        
        # Apply mask
        masked_error = abs_error * mask
        
        if self.reduction == 'none':
            return masked_error
        elif self.reduction == 'sum':
            return masked_error.sum()
        else:  # mean
            num_elements = mask.sum().clamp(min=1)
            return masked_error.sum() / num_elements


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features.
    
    Compares high-level features rather than pixel values,
    which often produces more visually pleasing results.
    """
    
    def __init__(
        self,
        layers: Tuple[str, ...] = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
        weights: Optional[Tuple[float, ...]] = None,
        normalize: bool = True
    ):
        """
        Initialize perceptual loss.
        
        Args:
            layers: VGG layers to use for feature extraction
            weights: Weights for each layer (default: equal weights)
            normalize: Whether to normalize input to VGG expected range
        """
        super().__init__()
        
        self.layers = layers
        self.weights = weights or tuple([1.0] * len(layers))
        self.normalize = normalize
        
        # Load VGG (lazy loading)
        self._vgg = None
        self._layer_indices = None
    
    @property
    def vgg(self):
        """Lazy load VGG model."""
        if self._vgg is None:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self._vgg = vgg
            
            # Map layer names to indices
            self._layer_indices = {
                'relu1_1': 1, 'relu1_2': 3,
                'relu2_1': 6, 'relu2_2': 8,
                'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
                'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
                'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35,
            }
        
        return self._vgg
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified VGG layers."""
        features = []
        target_indices = [self._layer_indices[l] for l in self.layers]
        
        # Move VGG to same device as input
        self.vgg.to(x.device)
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in target_indices:
                features.append(x)
        
        return features
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Optional mask (not typically used with perceptual loss)
            
        Returns:
            Loss value
        """
        # Convert grayscale to RGB if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize to ImageNet range
        if self.normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
            pred = (pred - mean) / std
            target = (target - mean) / std
        
        # Extract features
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        # Compute weighted feature loss
        loss = 0.0
        for w, pf, tf in zip(self.weights, pred_features, target_features):
            loss += w * F.mse_loss(pf, tf)
        
        return loss


class InpaintingLoss(nn.Module):
    """
    Combined loss for inpainting training.
    
    Combines:
    - Masked reconstruction loss (L1 or MSE)
    - Perceptual loss (optional)
    - Adversarial loss (optional, for GAN training)
    """
    
    def __init__(
        self,
        reconstruction_loss: str = 'l1',
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        use_perceptual: bool = True,
        mask_weight: float = 6.0,  # Weight for masked region vs non-masked
    ):
        """
        Initialize combined inpainting loss.
        
        Args:
            reconstruction_loss: 'l1' or 'mse'
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            use_perceptual: Whether to use perceptual loss
            mask_weight: Relative weight for masked region
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        self.mask_weight = mask_weight
        
        # Reconstruction loss
        if reconstruction_loss == 'l1':
            self.recon_loss = nn.L1Loss(reduction='none')
        else:
            self.recon_loss = nn.MSELoss(reduction='none')
        
        # Perceptual loss
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            mask: Binary mask [B, 1, H, W] (1 = inpaint region)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = 0.0
        
        # Reconstruction loss with mask weighting
        recon = self.recon_loss(pred, target)
        
        # Weight masked region more heavily
        weights = torch.ones_like(mask)
        weights = weights + (self.mask_weight - 1) * mask
        
        weighted_recon = (recon * weights).mean()
        loss_dict['reconstruction'] = weighted_recon.item()
        total_loss += self.reconstruction_weight * weighted_recon
        
        # Perceptual loss
        if self.use_perceptual:
            percep = self.perceptual_loss(pred, target)
            loss_dict['perceptual'] = percep.item()
            total_loss += self.perceptual_weight * percep
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion model training.
    
    Computes the noise prediction loss used in denoising diffusion.
    """
    
    def __init__(
        self,
        prediction_type: str = 'epsilon',  # 'epsilon' or 'v_prediction'
        loss_type: str = 'mse',  # 'mse' or 'l1'
        mask_weight: float = 1.0
    ):
        """
        Initialize diffusion loss.
        
        Args:
            prediction_type: What the model predicts ('epsilon' for noise)
            loss_type: Type of loss function
            mask_weight: Weight for masked region (1.0 = equal weight)
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        self.mask_weight = mask_weight
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            self.loss_fn = nn.L1Loss(reduction='none')
    
    def forward(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            model_output: Model's prediction (noise or velocity)
            target: Target value (noise that was added)
            mask: Optional mask for weighted loss
            
        Returns:
            Loss value
        """
        loss = self.loss_fn(model_output, target)
        
        if mask is not None and self.mask_weight != 1.0:
            # Weight masked region differently
            weights = torch.ones_like(loss)
            if mask.shape != loss.shape:
                mask = F.interpolate(mask, size=loss.shape[-2:], mode='nearest')
            weights = weights + (self.mask_weight - 1) * mask
            loss = loss * weights
        
        return loss.mean()
