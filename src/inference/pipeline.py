"""
Complete inference pipeline for converting cardiomegaly X-rays to healthy.

Orchestrates classifier, segmenter, inpainter, and validator components.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from ..config import Config, load_config, resolve_path


class CardiomegalyToHealthyPipeline:
    """
    Complete pipeline for converting cardiomegaly X-rays to healthy.
    
    Steps:
    1. Verify input is cardiomegaly (optional)
    2. Generate heart mask using segmenter
    3. Generate healthy heart using inpainter
    4. Validate output anatomically and with classifier
    5. Return best valid candidate
    """
    
    def __init__(
        self,
        classifier=None,
        segmenter=None,
        inpainter=None,
        validator=None,
        config: Optional[Config] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            classifier: CardiomegalyClassifier instance
            segmenter: ChexMaskSegmenter instance
            inpainter: CardiacInpainter instance
            validator: AnatomicalValidator instance
            config: Configuration object
        """
        self.classifier = classifier
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.validator = validator
        self.config = config or Config()
        
        # Default settings
        self.num_candidates = 5
        self.max_attempts = 10
        self.min_confidence = 0.80
        self.require_classifier = True
        self.require_anatomical = True
        
        if config:
            self._load_settings_from_config()
    
    def _load_settings_from_config(self):
        """Load settings from configuration."""
        if hasattr(self.config, 'inference'):
            inf = self.config.inference
            self.num_candidates = inf.get('num_candidates', 5)
            self.max_attempts = inf.get('max_attempts', 10)
        
        if hasattr(self.config, 'validation'):
            val = self.config.validation
            self.min_confidence = val.get('min_healthy_confidence', 0.80)
            self.require_classifier = val.get('require_classifier_validation', True)
            self.require_anatomical = val.get('require_anatomical_validation', True)
    
    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "CardiomegalyToHealthyPipeline":
        """
        Create pipeline from configuration file.
        
        Args:
            config_path: Path to configuration YAML file
            
        Returns:
            Initialized pipeline
        """
        from ..models.classifier import CardiomegalyClassifier
        from ..models.segmenter import ChexMaskSegmenter
        from ..models.inpainter import CardiacInpainter
        from ..validation.anatomical import AnatomicalValidator
        
        config = load_config(config_path)
        
        # Determine device
        device = "cuda" if config.device.get('cuda', True) else "cpu"
        
        # Initialize components
        classifier = None
        if hasattr(config, 'models') and config.models.get('classifier_path'):
            classifier_path = resolve_path(config.models.classifier_path)
            if classifier_path.exists():
                classifier = CardiomegalyClassifier(
                    weights_path=classifier_path,
                    device=device
                )
        
        segmenter = None
        if hasattr(config, 'models') and config.models.get('chexmask_weights_dir'):
            weights_dir = resolve_path(config.models.chexmask_weights_dir)
            if weights_dir.exists():
                segmenter = ChexMaskSegmenter(
                    weights_dir=weights_dir,
                    device=device
                )
        
        inpainter = CardiacInpainter(
            base_model_id=config.models.get('base_inpainting_model', 'runwayml/stable-diffusion-inpainting'),
            lora_weights_path=config.models.get('lora_weights_path'),
            device=device
        )
        
        validator = AnatomicalValidator(
            segmenter=segmenter,
            min_ctr=config.validation.get('min_ctr', 0.35),
            max_ctr=config.validation.get('max_ctr', 0.50)
        )
        
        return cls(
            classifier=classifier,
            segmenter=segmenter,
            inpainter=inpainter,
            validator=validator,
            config=config
        )
    
    def process(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        max_attempts: Optional[int] = None,
        return_all_candidates: bool = False,
        skip_verification: bool = False
    ) -> Union[Image.Image, List[Dict], None]:
        """
        Process a single cardiomegaly image.
        
        Args:
            image: Input chest X-ray (PIL Image, numpy array, or path)
            max_attempts: Maximum generation attempts (overrides config)
            return_all_candidates: If True, return all valid candidates
            skip_verification: If True, skip input cardiomegaly verification
            
        Returns:
            Best generated image, list of candidates (if return_all_candidates),
            or None if all attempts fail
        """
        max_attempts = max_attempts or self.max_attempts
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('L')
        elif isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image, mode='L')
        
        # Step 1: Verify input is cardiomegaly (optional)
        if not skip_verification and self.classifier is not None:
            pred, conf = self.classifier.predict_with_confidence(image)
            if pred == 'healthy':
                print(f"Input already classified as healthy (confidence: {conf:.2f})")
                if return_all_candidates:
                    return [{'image': image, 'ctr': None, 'confidence': conf, 'original': True}]
                return image
        
        # Step 2: Generate heart mask
        if self.segmenter is None:
            raise ValueError("Segmenter required for mask generation")
        
        masks = self.segmenter.segment(image)
        heart_mask = masks['heart']
        lung_mask = masks['lungs']
        
        # Prepare inpainting mask
        inpaint_mask = self.segmenter.prepare_inpainting_mask(heart_mask)
        
        # Step 3 & 4: Generate and validate candidates
        valid_candidates = []
        attempts = 0
        
        # Get prompts from config
        prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
        negative_prompt = "enlarged heart, cardiomegaly, artifacts, blur, noise"
        
        if hasattr(self.config, 'prompts'):
            prompt = self.config.prompts.get('generation', prompt)
            negative_prompt = self.config.prompts.get('negative', negative_prompt)
        
        while attempts < max_attempts and (
            len(valid_candidates) < self.num_candidates
        ):
            # Generate candidate
            try:
                candidates = self.inpainter.generate(
                    image=image,
                    mask=inpaint_mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_samples=1,
                    seed=42 + attempts  # Vary seed for different results
                )
                candidate = candidates[0]
            except Exception as e:
                print(f"Generation failed (attempt {attempts + 1}): {e}")
                attempts += 1
                continue
            
            # Convert to grayscale if needed
            if candidate.mode != 'L':
                candidate = candidate.convert('L')
            
            # Validate anatomically
            if self.require_anatomical and self.validator is not None:
                is_valid, details = self.validator.validate(
                    candidate,
                    heart_mask=None,  # Re-segment the generated image
                    lung_mask=None
                )
                
                if not is_valid:
                    attempts += 1
                    continue
                
                ctr = details.get('ctr')
            else:
                ctr = None
            
            # Validate with classifier
            classifier_conf = None
            if self.require_classifier and self.classifier is not None:
                pred, classifier_conf = self.classifier.predict_with_confidence(candidate)
                
                if pred != 'healthy' or classifier_conf < self.min_confidence:
                    attempts += 1
                    continue
            
            valid_candidates.append({
                'image': candidate,
                'ctr': ctr,
                'confidence': classifier_conf,
                'attempt': attempts
            })
            
            # Early exit if we have a great candidate
            if classifier_conf and classifier_conf > 0.95:
                break
            
            attempts += 1
        
        if not valid_candidates:
            print(f"Failed to generate valid image after {max_attempts} attempts")
            return None
        
        if return_all_candidates:
            return valid_candidates
        
        # Return best candidate (highest classifier confidence or first if no classifier)
        if self.classifier is not None:
            best = max(valid_candidates, key=lambda x: x['confidence'] or 0)
        else:
            best = valid_candidates[0]
        
        return best['image']
    
    def process_with_details(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        **kwargs
    ) -> Dict:
        """
        Process image and return detailed results.
        
        Args:
            image: Input image
            **kwargs: Additional arguments for process()
            
        Returns:
            Dictionary with full processing details
        """
        # Load original image
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert('L')
            image_path = str(image)
        else:
            original = image
            image_path = None
        
        # Get input classification
        input_classification = None
        if self.classifier is not None:
            pred, conf = self.classifier.predict_with_confidence(original)
            input_classification = {'prediction': pred, 'confidence': conf}
        
        # Get input CTR
        input_ctr = None
        if self.segmenter is not None and self.validator is not None:
            try:
                masks = self.segmenter.segment(original)
                input_ctr = self.validator.calculate_ctr(
                    original,
                    heart_mask=masks['heart'],
                    lung_mask=masks['lungs']
                )
            except:
                pass
        
        # Process
        kwargs['return_all_candidates'] = True
        result = self.process(image, **kwargs)
        
        if result is None:
            return {
                'success': False,
                'input_path': image_path,
                'input_classification': input_classification,
                'input_ctr': input_ctr,
                'output': None,
                'candidates': []
            }
        
        # Find best result
        if isinstance(result, list) and len(result) > 0:
            if result[0].get('original'):
                best = result[0]
            elif self.classifier is not None:
                best = max(result, key=lambda x: x['confidence'] or 0)
            else:
                best = result[0]
        else:
            best = result
        
        return {
            'success': True,
            'input_path': image_path,
            'input_classification': input_classification,
            'input_ctr': input_ctr,
            'output': best['image'],
            'output_ctr': best.get('ctr'),
            'output_confidence': best.get('confidence'),
            'candidates': result,
            'num_candidates': len(result) if isinstance(result, list) else 1
        }


class SimplePipeline:
    """
    Simplified pipeline for quick testing without all components.
    
    Uses only segmentation and inpainting (no classification validation).
    """
    
    def __init__(
        self,
        segmenter,
        inpainter,
        device: str = "cuda"
    ):
        """
        Initialize simple pipeline.
        
        Args:
            segmenter: ChexMaskSegmenter instance
            inpainter: CardiacInpainter instance
            device: Device to run on
        """
        self.segmenter = segmenter
        self.inpainter = inpainter
        self.device = device
    
    def process(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        prompt: str = "healthy chest xray normal heart",
        num_samples: int = 1
    ) -> List[Image.Image]:
        """
        Process an image.
        
        Args:
            image: Input image
            prompt: Generation prompt
            num_samples: Number of samples to generate
            
        Returns:
            List of generated images
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('L')
        
        # Get mask
        masks = self.segmenter.segment(image)
        inpaint_mask = self.segmenter.prepare_inpainting_mask(masks['heart'])
        
        # Generate
        results = self.inpainter.generate(
            image=image,
            mask=inpaint_mask,
            prompt=prompt,
            num_samples=num_samples
        )
        
        # Convert to grayscale
        return [r.convert('L') for r in results]
