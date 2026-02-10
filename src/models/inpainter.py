"""
Cardiac Inpainter using Stable Diffusion.

Implements the inpainting model for generating healthy hearts in chest X-rays.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class CardiacInpainter:
    """
    Stable Diffusion Inpainting model fine-tuned for chest X-rays.
    
    Base model: runwayml/stable-diffusion-inpainting
    Fine-tuning: LoRA adapters for efficient training
    """
    
    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-inpainting",
        lora_weights_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        use_half_precision: bool = True
    ):
        """
        Initialize the inpainter.
        
        Args:
            base_model_id: Hugging Face model ID for base inpainting model
            lora_weights_path: Path to LoRA adapter weights (optional)
            device: Device to run inference on
            use_half_precision: Use FP16 for faster inference
        """
        self.base_model_id = base_model_id
        self.lora_weights_path = Path(lora_weights_path) if lora_weights_path else None
        self.device = device
        self.dtype = torch.float16 if use_half_precision else torch.float32
        
        # Lazy loading of the pipeline
        self._pipeline = None
    
    @property
    def pipeline(self):
        """Lazy load the diffusion pipeline."""
        if self._pipeline is None:
            self._pipeline = self._load_pipeline()
        return self._pipeline
    
    def _load_pipeline(self):
        """Load the Stable Diffusion inpainting pipeline."""
        try:
            from diffusers import StableDiffusionInpaintPipeline
        except ImportError:
            raise ImportError(
                "diffusers is required for the inpainting model. "
                "Install with: pip install diffusers"
            )
        
        # Load base pipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            safety_checker=None,  # Disable for medical images
            requires_safety_checker=False
        )
        
        # Load LoRA weights if available
        if self.lora_weights_path and self.lora_weights_path.exists():
            self._load_lora_weights(pipeline)
        
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Enable memory optimizations
        try:
            pipeline.enable_attention_slicing()
        except:
            pass
        
        return pipeline
    
    def _load_lora_weights(self, pipeline):
        """Load LoRA weights into the pipeline."""
        lora_path = self.lora_weights_path
        
        # Check for different weight formats
        # 1. PEFT format (directory with adapter_config.json)
        # 2. State dict format (model.pt file)
        # 3. Checkpoint directory with lora_weights subdirectory
        
        peft_config_path = lora_path / 'adapter_config.json'
        lora_subdir = lora_path / 'lora_weights'
        model_pt_path = lora_path / 'model.pt'
        
        try:
            from peft import PeftModel, LoraConfig, get_peft_model
            
            if peft_config_path.exists():
                # Standard PEFT format
                pipeline.unet = PeftModel.from_pretrained(
                    pipeline.unet,
                    lora_path
                )
                print(f"Loaded LoRA weights (PEFT format) from {lora_path}")
                
            elif lora_subdir.exists() and (lora_subdir / 'adapter_config.json').exists():
                # Checkpoint with lora_weights subdirectory
                pipeline.unet = PeftModel.from_pretrained(
                    pipeline.unet,
                    lora_subdir
                )
                print(f"Loaded LoRA weights from {lora_subdir}")
                
            elif model_pt_path.exists():
                # State dict format - need to recreate PEFT model first
                print(f"Loading LoRA weights from state_dict: {model_pt_path}")
                
                # Get LoRA config (use defaults if not stored)
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    target_modules=['to_q', 'to_v', 'to_k', 'to_out.0'],
                    bias='none'
                )
                
                # Apply LoRA to UNet
                pipeline.unet = get_peft_model(pipeline.unet, lora_config)
                
                # Load state dict
                state_dict = torch.load(model_pt_path, map_location=self.device)
                pipeline.unet.load_state_dict(state_dict)
                print(f"Loaded LoRA weights (state_dict) from {model_pt_path}")
                
            else:
                print(f"Warning: No valid LoRA weights found at {lora_path}")
                print(f"  Checked: {peft_config_path}, {lora_subdir}, {model_pt_path}")
                
        except Exception as e:
            print(f"Error loading LoRA weights: {e}")
            import traceback
            traceback.print_exc()
    
    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        target_size: int = 512
    ) -> Image.Image:
        """
        Preprocess image for inpainting.
        
        Args:
            image: Input image
            target_size: Target size for the model
            
        Returns:
            Preprocessed PIL Image (RGB)
        """
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale [H, W] -> RGB [H, W, 3]
                image = np.stack([image] * 3, axis=-1)
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        return image
    
    def preprocess_mask(
        self,
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        target_size: int = 512
    ) -> Image.Image:
        """
        Preprocess mask for inpainting.
        
        Args:
            mask: Input mask (1 = inpaint region, 0 = keep)
            target_size: Target size for the model
            
        Returns:
            Preprocessed PIL Image (grayscale)
        """
        # Convert to numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask.squeeze(0)
        elif isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Ensure float [0, 1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Convert to uint8 for PIL
        mask = (mask * 255).astype(np.uint8)
        
        # Convert to PIL Image
        mask = Image.fromarray(mask, mode='L')
        
        # Resize
        if mask.size != (target_size, target_size):
            mask = mask.resize((target_size, target_size), Image.Resampling.NEAREST)
        
        return mask
    
    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: str = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality",
        negative_prompt: str = "enlarged heart, cardiomegaly, artifacts, blur, noise",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_samples: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate inpainted image(s).
        
        Args:
            image: Original image with cardiomegaly
            mask: Inpainting mask (1 = inpaint, 0 = keep)
            prompt: Text conditioning for generation
            negative_prompt: Text to avoid in generation
            num_inference_steps: Denoising steps (more = better quality)
            guidance_scale: Prompt adherence (7-8 typical)
            num_samples: Number of candidates to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of generated images
        """
        # Preprocess inputs
        image = self.preprocess_image(image)
        mask = self.preprocess_mask(mask)
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate
        results = []
        
        for i in range(num_samples):
            # Update generator for each sample if using seed
            if seed is not None and i > 0:
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
            
            output = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=1
            )
            
            results.append(output.images[0])
        
        return results
    
    def generate_single(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        **kwargs
    ) -> Image.Image:
        """
        Generate a single inpainted image.
        
        Args:
            image: Original image
            mask: Inpainting mask
            **kwargs: Additional arguments for generate()
            
        Returns:
            Single generated image
        """
        results = self.generate(image, mask, num_samples=1, **kwargs)
        return results[0]
    
    def to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert RGB output back to grayscale.
        
        Args:
            image: RGB image from the model
            
        Returns:
            Grayscale image
        """
        return image.convert('L')
    
    def postprocess(
        self,
        generated: Image.Image,
        original: Union[Image.Image, np.ndarray],
        mask: Union[np.ndarray, Image.Image],
        blend_edges: bool = True
    ) -> Image.Image:
        """
        Postprocess generated image.
        
        - Converts to grayscale
        - Blends with original at mask edges
        
        Args:
            generated: Generated image from the model
            original: Original input image
            mask: Inpainting mask used
            blend_edges: If True, blend edges for smoother transition
            
        Returns:
            Postprocessed image
        """
        # Convert to grayscale
        generated_gray = self.to_grayscale(generated)
        
        # Convert original to grayscale PIL if needed
        if isinstance(original, np.ndarray):
            if original.max() <= 1.0:
                original = (original * 255).astype(np.uint8)
            if original.ndim == 2:
                original = Image.fromarray(original, mode='L')
            else:
                original = Image.fromarray(original).convert('L')
        elif isinstance(original, Image.Image):
            original = original.convert('L')
        
        # Resize original to match generated
        original = original.resize(generated_gray.size, Image.Resampling.LANCZOS)
        
        # Get mask as array
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask).astype(np.float32)
        else:
            mask_array = mask.astype(np.float32)
        
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0
        
        # Resize mask to match
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray((mask_array * 255).astype(np.uint8))
        mask_img = mask_img.resize(generated_gray.size, PILImage.Resampling.LANCZOS)
        mask_array = np.array(mask_img).astype(np.float32) / 255.0
        
        if blend_edges:
            # Smooth the mask for blending
            import cv2
            mask_array = cv2.GaussianBlur(mask_array, (21, 21), 0)
        
        # Blend: generated in mask region, original outside
        gen_array = np.array(generated_gray).astype(np.float32)
        orig_array = np.array(original).astype(np.float32)
        
        blended = mask_array * gen_array + (1 - mask_array) * orig_array
        blended = blended.astype(np.uint8)
        
        return Image.fromarray(blended, mode='L')


class CardiacInpainterLite:
    """
    Lightweight inpainter for testing without Stable Diffusion.
    
    Uses simple image processing techniques instead of ML.
    Useful for pipeline testing and as a baseline.
    """
    
    def __init__(self, device: str = "cpu"):
        """Initialize the lite inpainter."""
        self.device = device
    
    def generate(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        shrink_factor: float = 0.85,
        num_samples: int = 1,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate inpainted images using simple shrinking approach.
        
        Args:
            image: Original image
            mask: Heart mask
            shrink_factor: How much to shrink the heart
            num_samples: Number of variations to generate
            **kwargs: Ignored (for API compatibility)
            
        Returns:
            List of generated images
        """
        import cv2
        
        # Convert inputs to numpy
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert('L')).astype(np.float32) / 255.0
        else:
            img_array = image.astype(np.float32)
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
        
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask).astype(np.float32) / 255.0
        else:
            mask_array = mask.astype(np.float32)
            if mask_array.max() > 1.0:
                mask_array = mask_array / 255.0
        
        results = []
        
        for i in range(num_samples):
            # Vary shrink factor slightly for each sample
            factor = shrink_factor - 0.05 * i / max(1, num_samples - 1)
            
            # Simple approach: inpaint with surrounding texture
            result = cv2.inpaint(
                (img_array * 255).astype(np.uint8),
                (mask_array * 255).astype(np.uint8),
                inpaintRadius=10,
                flags=cv2.INPAINT_TELEA
            )
            
            results.append(Image.fromarray(result, mode='L'))
        
        return results
