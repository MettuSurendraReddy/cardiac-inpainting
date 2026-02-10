"""
Training loop for the cardiac inpainting model.

Implements LoRA fine-tuning of Stable Diffusion for chest X-ray inpainting.
"""

import os
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .losses import DiffusionLoss, MaskedMSELoss


class InpaintingTrainer:
    """
    Training loop for the inpainting model with LoRA fine-tuning.
    
    Trains on healthy chest X-rays to learn the distribution of
    healthy hearts in context.
    """
    
    def __init__(
        self,
        model,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        device: str = "cuda",
        noise_scheduler=None,
        vae=None,
        tokenizer=None,
        text_encoder=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The inpainting model (or UNet for diffusion)
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            config: Training configuration dictionary
            device: Device to train on
            noise_scheduler: Diffusion noise scheduler
            vae: VAE for encoding/decoding images to latent space
            tokenizer: CLIP tokenizer for text encoding
            text_encoder: CLIP text encoder for generating embeddings from prompts
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        
        # Move VAE to device and freeze
        if self.vae is not None:
            self.vae = self.vae.to(device)
            self.vae.requires_grad_(False)
        
        # Move text_encoder to device and freeze
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder.requires_grad_(False)
        
        # Default config
        self.config = {
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'warmup_steps': 500,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0,
            'mixed_precision': True,
            'save_every_epochs': 10,
            'validate_every_epochs': 5,
            'log_every_steps': 100,
            'output_dir': 'models/inpainting',
            'early_stopping_patience': 10,
        }
        if config:
            self.config.update(config)
        
        # Setup
        self._setup_training()
    
    def _setup_training(self):
        """Setup optimizer, scheduler, scaler, etc."""
        # Optimizer - only trainable parameters (LoRA weights)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler with warmup
        num_training_steps = len(self.train_dataloader) * self.config['num_epochs']
        warmup_steps = self.config['warmup_steps']
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=self.config['learning_rate'] * 0.1
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
        
        # Loss function
        self.loss_fn = DiffusionLoss(mask_weight=2.0)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.train_losses = []
        self.val_losses = []
    
    def train(self) -> Dict:
        """
        Run the full training loop.
        
        Returns:
            Dictionary with training history
        """
        start_epoch = self.epoch
        print(f"Starting training for {self.config['num_epochs']} epochs")
        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.epoch = epoch
            
            # Train one epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']} - Train Loss: {train_loss:.6f}")
            
            # Validation
            if self.val_dataloader and (epoch + 1) % self.config['validate_every_epochs'] == 0:
                val_loss = self._validate()
                self.val_losses.append(val_loss)
                print(f"  Validation Loss: {val_loss:.6f}")
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint('best_model')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['early_stopping_patience']:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.config['save_every_epochs'] == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
        
        # Save final model
        self._save_checkpoint('final_model')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epoch + 1
        }
    
    def _train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self._train_step(batch)
            
            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation_steps']
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['max_grad_norm']
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            epoch_loss += loss.item() * self.config['gradient_accumulation_steps']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * self.config['gradient_accumulation_steps']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return epoch_loss / num_batches
    
    def _train_step(self, batch: Dict) -> torch.Tensor:
        """
        Single training step.
        
        Args:
            batch: Dictionary with 'image', 'mask', 'masked_image'
            
        Returns:
            Loss value
        """
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
            batch_size = images.shape[0]
            
            # Encode images to latent space using VAE
            if self.vae is not None:
                # Convert grayscale to RGB if needed (SD expects 3 channels)
                if images.shape[1] == 1:
                    images_rgb = images.repeat(1, 3, 1, 1)
                else:
                    images_rgb = images
                
                # Encode to latent space
                latents = self.vae.encode(images_rgb).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                
                # Resize mask to latent size
                latent_size = latents.shape[-1]
                masks_latent = F.interpolate(masks, size=(latent_size, latent_size), mode='nearest')
                
                # Create masked latents
                masked_images_rgb = images_rgb * (1 - masks)
                masked_latents = self.vae.encode(masked_images_rgb).latent_dist.sample()
                masked_latents = masked_latents * self.vae.config.scaling_factor
            else:
                latents = images
                masks_latent = masks
                masked_latents = images * (1 - masks)
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps if self.noise_scheduler else 1000,
                (batch_size,), device=self.device, dtype=torch.long
            )
            
            # Sample noise
            noise = torch.randn_like(latents)
            
            # Add noise using scheduler
            if self.noise_scheduler is not None:
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            else:
                alpha = self._get_alpha(timesteps)
                noisy_latents = alpha.sqrt() * latents + (1 - alpha).sqrt() * noise
            
            # Concatenate inputs for inpainting model
            # SD inpainting UNet expects 9 channels: [noisy_latent(4), mask(1), masked_latent(4)]
            model_input = torch.cat([noisy_latents, masks_latent, masked_latents], dim=1)
            
            # Create encoder hidden states from text prompt
            encoder_hidden_states = self._get_encoder_hidden_states(batch_size)
            
            # Predict noise
            noise_pred = self.model(
                model_input, 
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Compute loss ONLY in masked region (as per project.md)
            # This focuses the model on learning to generate good content in the inpainted area
            loss = F.mse_loss(noise_pred * masks_latent, noise * masks_latent)
        
        return loss
    
    def _get_encoder_hidden_states(self, batch_size: int) -> torch.Tensor:
        """Get encoder hidden states from text prompts."""
        # Use prompts from project.md
        prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
        
        if self.tokenizer is None or self.text_encoder is None:
            # Fallback to unconditional if tokenizer/text_encoder not provided
            if not hasattr(self, '_cached_encoder_states'):
                cross_attention_dim = getattr(self.model.config, 'cross_attention_dim', 768)
                self._cached_encoder_states = torch.zeros(
                    1, 77, cross_attention_dim, 
                    device=self.device, dtype=torch.float16 if self.config['mixed_precision'] else torch.float32
                )
            return self._cached_encoder_states.expand(batch_size, -1, -1)
        
        # Tokenize prompt
        text_inputs = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]
        
        return encoder_hidden_states
    
    def _get_alpha(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get alpha values for given timesteps (simplified noise schedule)."""
        # Linear schedule (simplified)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, 1000, device=self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        return alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    
    @torch.no_grad()
    def _validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            batch_size = images.shape[0]
            
            with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
                # Encode images to latent space using VAE
                if self.vae is not None:
                    # Convert grayscale to RGB if needed
                    if images.shape[1] == 1:
                        images_rgb = images.repeat(1, 3, 1, 1)
                    else:
                        images_rgb = images
                    
                    # Encode to latent space
                    latents = self.vae.encode(images_rgb).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # Resize mask to latent size
                    latent_size = latents.shape[-1]
                    masks_latent = F.interpolate(masks, size=(latent_size, latent_size), mode='nearest')
                    
                    # Create masked latents
                    masked_images_rgb = images_rgb * (1 - masks)
                    masked_latents = self.vae.encode(masked_images_rgb).latent_dist.sample()
                    masked_latents = masked_latents * self.vae.config.scaling_factor
                else:
                    latents = images
                    masks_latent = masks
                    masked_latents = images * (1 - masks)
                
                # Sample timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps if self.noise_scheduler else 1000,
                    (batch_size,), device=self.device, dtype=torch.long
                )
                
                # Sample noise
                noise = torch.randn_like(latents)
                
                # Add noise
                if self.noise_scheduler is not None:
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                else:
                    alpha = self._get_alpha(timesteps)
                    noisy_latents = alpha.sqrt() * latents + (1 - alpha).sqrt() * noise
                
                # Create input
                model_input = torch.cat([noisy_latents, masks_latent, masked_latents], dim=1)
                
                # Get encoder hidden states
                encoder_hidden_states = self._get_encoder_hidden_states(batch_size)
                
                # Predict
                noise_pred = self.model(
                    model_input, 
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # Loss
                loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state - check if it's a PEFT model
        try:
            # For PEFT/LoRA models, use save_pretrained
            if hasattr(self.model, 'save_pretrained'):
                lora_dir = checkpoint_dir / 'lora_weights'
                self.model.save_pretrained(lora_dir)
                print(f"  Saved LoRA weights to {lora_dir}")
            else:
                # Regular model - save state_dict
                torch.save(
                    self.model.state_dict(),
                    checkpoint_dir / 'model.pt'
                )
        except Exception as e:
            # Fallback to state_dict
            print(f"  Warning: Could not save pretrained ({e}), using state_dict")
            torch.save(
                self.model.state_dict(),
                checkpoint_dir / 'model.pt'
            )
        
        # Save optimizer and scheduler state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, checkpoint_dir / 'training_state.pt')
        
        print(f"Saved checkpoint: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load a training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model state
        model_path = checkpoint_path / 'model.pt'
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path))
        
        # Load training state
        state_path = checkpoint_path / 'training_state.pt'
        if state_path.exists():
            state = torch.load(state_path)
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.epoch = state['epoch']
            self.global_step = state['global_step']
            self.best_val_loss = state['best_val_loss']
        
        print(f"Loaded checkpoint from {checkpoint_path}")


class LoRATrainer(InpaintingTrainer):
    """
    Specialized trainer for LoRA fine-tuning of Stable Diffusion.
    
    Uses the PEFT library for efficient LoRA training.
    """
    
    def __init__(
        self,
        unet,
        noise_scheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        lora_config: Optional[Dict] = None,
        config: Optional[Dict] = None,
        device: str = "cuda"
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            unet: The UNet model from Stable Diffusion
            noise_scheduler: The noise scheduler from diffusers
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            lora_config: LoRA configuration
            config: Training configuration
            device: Device to train on
        """
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.device = device
        
        # Apply LoRA
        self._apply_lora(lora_config or {})
        
        # Initialize base trainer with LoRA model
        super().__init__(
            model=self.unet,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            device=device
        )
    
    def _apply_lora(self, lora_config: Dict):
        """Apply LoRA adapters to the UNet."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError("PEFT library required for LoRA training. Install with: pip install peft")
        
        # Default LoRA config
        default_lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': ['to_q', 'to_v', 'to_k', 'to_out.0'],
            'bias': 'none'
        }
        default_lora_config.update(lora_config)
        
        # Create PEFT config
        peft_config = LoraConfig(
            r=default_lora_config['r'],
            lora_alpha=default_lora_config['lora_alpha'],
            lora_dropout=default_lora_config['lora_dropout'],
            target_modules=default_lora_config['target_modules'],
            bias=default_lora_config['bias']
        )
        
        # Apply LoRA
        self.unet = get_peft_model(self.unet, peft_config)
        self.unet.print_trainable_parameters()
    
    def _train_step(self, batch: Dict) -> torch.Tensor:
        """
        Training step using diffusers noise scheduler.
        
        Args:
            batch: Dictionary with 'image', 'mask', 'masked_image'
            
        Returns:
            Loss value
        """
        images = batch['image'].to(self.device)
        masks = batch['mask'].to(self.device)
        masked_images = batch['masked_image'].to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.config['mixed_precision']):
            batch_size = images.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=self.device, dtype=torch.long
            )
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Add noise using scheduler
            noisy_images = self.noise_scheduler.add_noise(images, noise, timesteps)
            
            # Prepare UNet input (for inpainting: concatenate mask and masked image)
            # Standard SD inpainting expects [latent, mask, masked_latent]
            # For image space: [noisy_image, mask, masked_image]
            unet_input = torch.cat([noisy_images, masks, masked_images], dim=1)
            
            # Create dummy encoder hidden states (unconditional generation)
            # UNet expects encoder_hidden_states from text encoder
            # For unconditional training, use zeros or cached empty prompt embedding
            encoder_hidden_states = self._get_encoder_hidden_states(batch_size)
            
            # Predict noise
            noise_pred = self.unet(
                unet_input, 
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    def _get_encoder_hidden_states(self, batch_size: int) -> torch.Tensor:
        """Get encoder hidden states for unconditional generation."""
        # SD inpainting uses 77 tokens with 768 dim (SD 1.x) or 1024 dim (SD 2.x)
        # For unconditional training, we use zeros
        if not hasattr(self, '_cached_encoder_states'):
            # Try to get the correct dimension from UNet config
            cross_attention_dim = self.unet.config.cross_attention_dim
            self._cached_encoder_states = torch.zeros(
                1, 77, cross_attention_dim, 
                device=self.device, dtype=torch.float16 if self.config['mixed_precision'] else torch.float32
            )
        return self._cached_encoder_states.expand(batch_size, -1, -1)
    
    def save_lora_weights(self, save_path: Union[str, Path]):
        """Save only the LoRA weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.unet.save_pretrained(save_path)
        print(f"Saved LoRA weights to {save_path}")
