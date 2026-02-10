#!/usr/bin/env python
"""
Training script for the cardiac inpainting model.

Fine-tunes a Stable Diffusion inpainting model with LoRA on healthy chest X-rays.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_path, Config


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the output directory."""
    if not output_dir.exists():
        return None
    
    # Look for checkpoint_epoch_* directories
    checkpoints = []
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith('checkpoint_epoch_'):
            try:
                epoch_num = int(path.name.split('_')[-1])
                checkpoints.append((epoch_num, path))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Return the checkpoint with highest epoch number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main():
    parser = argparse.ArgumentParser(description="Train the inpainting model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override processed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory for checkpoints"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic resume from latest checkpoint"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Override gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one batch only (for testing)"
    )
    
    args = parser.parse_args()

    is_cuda = str(args.device).startswith("cuda")
    if is_cuda:
        # Helps reduce fragmentation on long runs / tight VRAM.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Load configuration - merge training config with defaults
    project_root = Path(__file__).parent.parent
    defaults_path = project_root / "configs" / "default.yaml"
    
    from src.config import Config
    config = Config.from_yaml_with_defaults(args.config, defaults_path)
    
    # Override config with command line arguments
    if args.epochs:
        config.training._config['num_epochs'] = args.epochs
    if args.batch_size:
        config.training._config['batch_size'] = args.batch_size
    if args.grad_accum_steps:
        config.training._config['gradient_accumulation_steps'] = args.grad_accum_steps
    if args.learning_rate:
        config.training._config['learning_rate'] = args.learning_rate
    
    # Setup paths
    data_dir = args.data_dir or resolve_path(
        config.paths.processed_data_dir if hasattr(config, 'paths') else 'data/processed',
        project_root
    )
    output_dir = args.output_dir or resolve_path(
        'models/inpainting',
        project_root
    )
    
    # Auto-resume from latest checkpoint if available
    resume_from = args.resume
    if not resume_from and not args.no_auto_resume:
        latest_checkpoint = find_latest_checkpoint(output_dir)
        if latest_checkpoint:
            resume_from = latest_checkpoint
            print(f"\n🔄 Auto-resuming from: {latest_checkpoint.name}")
    
    print("Training Configuration")
    print("=" * 50)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Grad accum steps: {config.training.gradient_accumulation_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    if resume_from:
        print(f"Resume from: {resume_from}")
    print("=" * 50)
    
    # Check data exists
    train_images = data_dir / 'train' / 'images'
    train_masks = data_dir / 'train' / 'masks'
    
    if not train_images.exists() or not train_masks.exists():
        print(f"\nError: Training data not found at {data_dir}")
        print("Please run 'python scripts/prepare_data.py' first.")
        sys.exit(1)
    
    # Import required modules
    import torch
    from torch.utils.data import DataLoader
    
    from src.data.dataset import CardiacInpaintingDataset, create_dataloaders
    from src.data.augmentation import get_train_augmentations, get_val_augmentations
    
    # Create datasets
    print("\nLoading datasets...")
    
    train_transform = get_train_augmentations(
        image_size=config.image.size,
        rotation_limit=5,
        brightness_limit=0.1,
        contrast_limit=0.1
    )
    
    train_dataset = CardiacInpaintingDataset(
        images_dir=train_images,
        masks_dir=train_masks,
        image_size=config.image.size,
        transform=train_transform,
        dilate_mask_range=(0.3, 0.8)  # Dilate masks 30-80% to simulate cardiomegaly
    )
    print("  ✓ Training dataset with mask dilation (30-80%) enabled")
    
    val_images = data_dir / 'val' / 'images'
    val_masks = data_dir / 'val' / 'masks'
    
    val_dataset = None
    if val_images.exists() and val_masks.exists():
        val_dataset = CardiacInpaintingDataset(
            images_dir=val_images,
            masks_dir=val_masks,
            image_size=config.image.size,
            transform=get_val_augmentations(config.image.size)
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.data.get('num_workers', 4)
    ) if val_dataset else (
        DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.get('num_workers', 4)
        ),
        None
    )
    
    # Initialize model and trainer
    print("\nInitializing model...")
    
    # For LoRA training, we need to load the base model and apply LoRA
    try:
        from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
        from peft import LoraConfig, get_peft_model
        
        # Load base model
        base_model_id = config.models.get(
            'base_inpainting_model',
            'runwayml/stable-diffusion-inpainting'
        )
        
        print(f"Loading base model: {base_model_id}")
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if is_cuda else torch.float32
        )
        
        unet = pipe.unet
        noise_scheduler = pipe.scheduler
        vae = pipe.vae
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=list(config.lora.target_modules),
        )
        
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
        
    except ImportError as e:
        print(f"Error: Required libraries not installed: {e}")
        print("Install with: pip install diffusers peft accelerate")
        sys.exit(1)
    
    # Initialize trainer
    from src.training.trainer import InpaintingTrainer
    
    trainer_config = {
        'learning_rate': config.training.learning_rate,
        'num_epochs': config.training.num_epochs,
        'warmup_steps': config.training.warmup_steps,
        'gradient_accumulation_steps': config.training.gradient_accumulation_steps,
        'max_grad_norm': config.training.get('max_grad_norm', 1.0),
        'mixed_precision': config.training.mixed_precision,
        'save_every_epochs': config.training.save_every_epochs,
        'validate_every_epochs': config.training.validate_every_epochs,
        'output_dir': str(output_dir),
        'early_stopping_patience': config.training.early_stopping.get('patience', 10)
    }
    
    trainer = InpaintingTrainer(
        model=unet,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=trainer_config,
        device=args.device,
        noise_scheduler=noise_scheduler,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # Dry run mode
    if args.dry_run:
        print("\n[DRY RUN] Running one batch only...")
        batch = next(iter(train_loader))
        loss = trainer._train_step(batch)
        print(f"Dry run loss: {loss.item():.4f}")
        print("Dry run complete!")
        return
    
    # Initialize wandb if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config.logging.wandb.get('project', 'cardiac-inpainting'),
                config=trainer_config
            )
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
    
    # Train
    print("\nStarting training...")
    try:
        history = trainer.train()
    except torch.OutOfMemoryError as e:
        if is_cuda:
            print("\n❌ CUDA out of memory أثناء träning.")
            print("Åtgärder som brukar lösa det på 8GB-kort:")
            print("- Stäng andra GPU-appar (Chrome/Discord/OBS, etc.) och kör igen")
            print("- Kör med lägre batch: `python scripts/train.py --batch-size 1`")
            print("- Om du vill behålla ungefär samma effektiva batch: `--grad-accum-steps 4` eller `8`")
            print("- Om det fortfarande OOM: prova `--batch-size 1 --grad-accum-steps 1` (snabb sanity)")
        raise e
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"  Epochs trained: {history['epochs_trained']}")
    print(f"  Best validation loss: {history['best_val_loss']:.6f}")
    print(f"  Checkpoints saved to: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
