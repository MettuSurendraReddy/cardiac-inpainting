"""Cardiomegaly classifier wrapper.

Loads a binary classifier (healthy vs cardiomegaly) trained on chest X-rays.

Note: The workspace may contain checkpoints trained with different torchvision
backbones (e.g. ResNet/EfficientNet/DenseNet). This wrapper auto-detects a
compatible architecture from the checkpoint state_dict.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


class CardiomegalyClassifier:
    """
    Wrapper for the pre-trained cardiomegaly classifier.
    
    This classifier distinguishes between:
    - Cardiomegaly (enlarged heart)
    - Healthy (normal heart)
    """
    
    def __init__(
        self,
        weights_path: Union[str, Path],
        device: str = "cuda",
        image_size: int = 224,
        temperature: float = 1.0,
    ):
        """
        Initialize the classifier.
        
        Args:
            weights_path: Path to the trained model weights
            device: Device to run inference on
            image_size: Input image size expected by the model
        """
        self.device = device
        self.image_size = image_size
        self.weights_path = Path(weights_path)
        self.temperature = float(temperature) if temperature is not None else 1.0
        if not (self.temperature > 0.0):
            raise ValueError("temperature must be > 0")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Class labels
        self.classes = ['healthy', 'cardiomegaly']
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _probs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities.

        Note: Softmax probabilities are commonly *overconfident* unless the model
        has been explicitly calibrated. Temperature scaling (T>1) can reduce
        overconfidence, but should ideally be fit on a validation set.
        """
        return F.softmax(logits / self.temperature, dim=1)
    
    def _load_model(self) -> nn.Module:
        """Load a compatible torchvision model from the checkpoint."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        # Load checkpoint/state_dict.
        try:
            state_obj = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # Older torch versions don't support weights_only.
            state_obj = torch.load(
                self.weights_path,
                map_location=self.device,
            )

        # Unwrap common checkpoint formats.
        state_dict = state_obj
        if isinstance(state_obj, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state_obj and isinstance(state_obj[key], dict):
                    state_dict = state_obj[key]
                    break

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"Invalid classifier checkpoint (no state_dict): {self.weights_path}")

        # Some training code saves keys with a module prefix.
        sample_keys = list(state_dict.keys())
        if sample_keys and all(k.startswith("module.") for k in sample_keys[: min(10, len(sample_keys))]):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

        def _try_build_and_load(build_fn) -> Optional[nn.Module]:
            m = build_fn()
            try:
                m.load_state_dict(state_dict, strict=True)
                return m
            except RuntimeError:
                return None

        # Candidate builders. Prefer matching by signature keys first.
        def build_resnet18():
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            return m

        def build_resnet34():
            m = models.resnet34(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            return m

        def build_resnet50():
            m = models.resnet50(weights=None)
            m.fc = nn.Linear(m.fc.in_features, 2)
            return m

        def build_efficientnet_b0():
            m = models.efficientnet_b0(weights=None)
            m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
            return m

        def build_densenet121():
            m = models.densenet121(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            return m

        def build_densenet169():
            m = models.densenet169(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            return m

        def build_densenet201():
            m = models.densenet201(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            return m

        def build_densenet161():
            m = models.densenet161(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, 2)
            return m

        keys_joined = " ".join(sample_keys[:200])
        prioritized_builders = []

        # Heuristics from key prefixes.
        if "features." in keys_joined and "denseblock" in keys_joined:
            prioritized_builders = [
                build_densenet121,
                build_densenet169,
                build_densenet201,
                build_densenet161,
            ]
        elif "classifier.0.weight" in keys_joined or "features.0.0.weight" in keys_joined:
            prioritized_builders = [
                build_efficientnet_b0,
                build_densenet121,
                build_densenet169,
                build_densenet201,
                build_densenet161,
            ]
        elif "layer1.0.conv1.weight" in keys_joined or "conv1.weight" in keys_joined:
            prioritized_builders = [build_resnet18, build_resnet34, build_resnet50]

        # Full fallback list if heuristics weren't decisive.
        if not prioritized_builders:
            prioritized_builders = [
                build_resnet18,
                build_resnet34,
                build_resnet50,
                build_efficientnet_b0,
                build_densenet121,
                build_densenet169,
                build_densenet201,
                build_densenet161,
            ]

        for build_fn in prioritized_builders:
            model = _try_build_and_load(build_fn)
            if model is not None:
                return model.to(self.device)

        # If nothing matched, raise a clear error.
        raise RuntimeError(
            "Could not load classifier weights: checkpoint architecture mismatch. "
            f"Checkpoint: {self.weights_path}"
        )
    
    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess an image for classification.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale to RGB
                image = np.stack([image] * 3, axis=-1)
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 1:
                # [1, H, W] -> [H, W]
                image = image.squeeze(0)
            image = image.cpu().numpy()
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]
    ) -> str:
        """
        Predict the class of an image.
        
        Args:
            image: Input image (PIL Image, numpy array, tensor, or path)
            
        Returns:
            Predicted class label ('healthy' or 'cardiomegaly')
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        
        return self.classes[pred_idx]
    
    @torch.no_grad()
    def predict_with_confidence(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]
    ) -> Tuple[str, float]:
        """
        Predict the class of an image with confidence score.
        
        Args:
            image: Input image (PIL Image, numpy array, tensor, or path)
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        probabilities = self._probs_from_logits(output)
        
        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[0, pred_idx].item()
        
        return self.classes[pred_idx], confidence
    
    @torch.no_grad()
    def predict_proba(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path]
    ) -> Dict[str, float]:
        """
        Get class probabilities for an image.
        
        Args:
            image: Input image (PIL Image, numpy array, tensor, or path)
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)
        
        # Forward pass
        output = self.model(input_tensor)
        probabilities = self._probs_from_logits(output)
        
        return {
            cls: probabilities[0, i].item()
            for i, cls in enumerate(self.classes)
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: list
    ) -> list:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of predicted class labels
        """
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Forward pass
        outputs = self.model(batch)
        pred_indices = outputs.argmax(dim=1).tolist()
        
        return [self.classes[idx] for idx in pred_indices]
    
    def is_cardiomegaly(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
        threshold: float = 0.5
    ) -> bool:
        """
        Check if an image shows cardiomegaly.
        
        Args:
            image: Input image
            threshold: Confidence threshold for cardiomegaly detection
            
        Returns:
            True if cardiomegaly detected, False otherwise
        """
        proba = self.predict_proba(image)
        return proba['cardiomegaly'] > threshold
    
    def is_healthy(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, str, Path],
        threshold: float = 0.5
    ) -> bool:
        """
        Check if an image shows a healthy heart.
        
        Args:
            image: Input image
            threshold: Confidence threshold for healthy classification
            
        Returns:
            True if healthy, False otherwise
        """
        proba = self.predict_proba(image)
        return proba['healthy'] > threshold
