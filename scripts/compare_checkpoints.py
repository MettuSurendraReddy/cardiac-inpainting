#!/usr/bin/env python
"""
Compare checkpoints across training epochs.

Tests multiple checkpoints and creates a visual comparison showing
how the model improves over training.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "CheXmask-Database" / "HybridGNet"))


def list_available_images(image_dir: Path) -> list:
    """List all available cardiomegaly images."""
    images = sorted(image_dir.glob("*.png"))
    return images


def load_segmenter(device):
    """Load the heart segmentation model."""
    from scripts.generate_masks import load_hybridgnet
    weights_path = PROJECT_ROOT / "models" / "CheXmask-Database" / "Weights" / "SegmentationModel" / "bestMSE.pt"
    return load_hybridgnet(weights_path, device)


def generate_heart_mask(segmenter, image_path, device):
    """Generate heart mask for an image."""
    from scripts.generate_masks import process_image
    
    # process_image expects (model, image_path, device)
    masks = process_image(segmenter, image_path, device)
    
    # Load original image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    return masks['heart'], image


def calculate_ctr(image, segmenter, device):
    """Calculate Cardiothoracic Ratio."""
    import tempfile
    import os
    from scripts.generate_masks import process_image
    
    if isinstance(image, Image.Image):
        image = np.array(image.convert('L'))
    
    # Save image temporarily for process_image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, image)
    
    try:
        masks = process_image(segmenter, Path(temp_path), device)
        
        # Get heart width from mask
        heart_mask = masks['heart']
        heart_cols = np.where(heart_mask.sum(axis=0) > 0)[0]
        if len(heart_cols) == 0:
            return None
        heart_width = heart_cols[-1] - heart_cols[0]
        
        # Get chest width from lung masks
        left_lung = masks['left_lung']
        right_lung = masks['right_lung']
        combined_lungs = np.maximum(left_lung, right_lung)
        lung_cols = np.where(combined_lungs.sum(axis=0) > 0)[0]
        if len(lung_cols) == 0:
            return None
        chest_width = lung_cols[-1] - lung_cols[0]
        
        if chest_width == 0:
            return None
        
        return heart_width / chest_width
    finally:
        os.unlink(temp_path)


def load_inpainting_model(checkpoint_path, device):
    """Load inpainting model from checkpoint."""
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel, LoraConfig, get_peft_model
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights
    lora_dir = checkpoint_path / "lora_weights"
    model_pt = checkpoint_path / "model.pt"
    
    if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    elif model_pt.exists():
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=['to_q', 'to_v', 'to_k', 'to_out.0'],
            bias='none'
        )
        pipe.unet = get_peft_model(pipe.unet, lora_config)
        state_dict = torch.load(model_pt, map_location=device)
        pipe.unet.load_state_dict(state_dict)
    else:
        return None
    
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def run_inpainting(pipe, img_pil, mask_pil, device, num_candidates=3):
    """Run inpainting and return best candidate."""
    prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
    negative_prompt = "enlarged heart, cardiomegaly, artifacts, blur, noise"
    
    candidates = []
    for i in range(num_candidates):
        with torch.autocast(device):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=torch.Generator(device).manual_seed(42 + i)
            ).images[0]
        candidates.append(result)
    
    return candidates[0]  # Return first candidate for consistency


def main():
    parser = argparse.ArgumentParser(description="Compare checkpoints across epochs")
    parser.add_argument(
        "--image",
        type=str,
        help="Path or filename of cardiomegaly image to test"
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="List all available cardiomegaly images"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="40,50,60,70,80,90,100",
        help="Comma-separated list of epochs to test (default: 40,50,60,70,80,90,100)"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=3,
        help="Number of candidates per checkpoint"
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_dir = PROJECT_ROOT / "data" / "raw" / "cardiomegaly"
    output_dir = PROJECT_ROOT / "outputs" / "checkpoint_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List images mode
    if args.list_images:
        print("\nAvailable cardiomegaly images:")
        print("=" * 50)
        images = list_available_images(image_dir)
        for i, img in enumerate(images[:20]):  # Show first 20
            print(f"  {i+1}. {img.name}")
        if len(images) > 20:
            print(f"  ... and {len(images) - 20} more")
        print(f"\nTotal: {len(images)} images")
        print("\nUsage: python scripts/compare_checkpoints.py --image <filename>")
        return
    
    # Parse epochs
    epochs = [int(e.strip()) for e in args.epochs.split(",")]
    
    # Select test image
    if args.image:
        if Path(args.image).exists():
            test_image_path = Path(args.image)
        else:
            test_image_path = image_dir / args.image
            if not test_image_path.exists():
                # Try with .png extension
                test_image_path = image_dir / f"{args.image}.png"
    else:
        images = list_available_images(image_dir)
        if not images:
            print("No cardiomegaly images found!")
            return
        test_image_path = images[0]
    
    if not test_image_path.exists():
        print(f"Image not found: {test_image_path}")
        return
    
    print("=" * 60)
    print("Checkpoint Comparison Tool")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Test image: {test_image_path.name}")
    print(f"Epochs to test: {epochs}")
    print("=" * 60)
    
    # Load segmenter
    print("\n[1/4] Loading segmenter...")
    segmenter = load_segmenter(device)
    print("  ✓ Segmenter loaded")
    
    # Generate mask
    print("\n[2/4] Generating heart mask...")
    heart_mask, original_img = generate_heart_mask(segmenter, test_image_path, device)
    original_size = original_img.shape[:2]
    print(f"  ✓ Mask generated (shape: {heart_mask.shape})")
    
    # Prepare image and mask for inpainting
    img_resized = cv2.resize(original_img, (512, 512))
    mask_resized = cv2.resize(heart_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    img_pil = Image.fromarray(cv2.cvtColor(
        np.stack([img_resized]*3, axis=-1), 
        cv2.COLOR_BGR2RGB
    ))
    
    # Dilate mask
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask_resized, kernel, iterations=2)
    mask_pil = Image.fromarray(mask_dilated)
    
    # Calculate original CTR
    original_ctr = calculate_ctr(original_img, segmenter, device)
    print(f"\n  Original CTR: {original_ctr:.3f}" if original_ctr else "  Original CTR: N/A")
    
    # Test each checkpoint
    print("\n[3/4] Testing checkpoints...")
    results = {}
    
    for epoch in epochs:
        checkpoint_path = PROJECT_ROOT / "models" / "inpainting" / f"checkpoint_epoch_{epoch}"
        
        if not checkpoint_path.exists():
            print(f"  ⚠ Epoch {epoch}: checkpoint not found, skipping")
            continue
        
        print(f"  Testing epoch {epoch}...", end=" ")
        
        try:
            pipe = load_inpainting_model(checkpoint_path, device)
            if pipe is None:
                print("✗ (no weights)")
                continue
            
            result = run_inpainting(pipe, img_pil, mask_pil, device, args.num_candidates)
            
            # Convert to grayscale and resize
            result_gray = np.array(result.convert('L'))
            result_resized = cv2.resize(result_gray, (original_size[1], original_size[0]))
            
            # Calculate CTR
            ctr = calculate_ctr(result_resized, segmenter, device)
            
            results[epoch] = {
                'image': result_gray,
                'image_full': result_resized,
                'ctr': ctr
            }
            
            status = "✓" if ctr and ctr < 0.5 else "○"
            ctr_str = f"CTR: {ctr:.3f}" if ctr else "CTR: N/A"
            print(f"{status} ({ctr_str})")
            
            # Clean up
            del pipe
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ (error: {e})")
    
    if not results:
        print("\nNo checkpoints could be tested!")
        return
    
    # Create comparison visualization
    print("\n[4/4] Creating comparison...")
    
    num_results = len(results) + 1  # +1 for original
    cols = min(4, num_results)
    rows = (num_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Hide all axes first
    for ax in axes:
        ax.axis('off')
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    ctr_str = f"CTR: {original_ctr:.3f}" if original_ctr else "CTR: N/A"
    axes[0].set_title(f"Original (Cardiomegaly)\n{ctr_str}", fontsize=10)
    axes[0].axis('off')
    
    # Results for each epoch
    for i, (epoch, data) in enumerate(sorted(results.items())):
        ax = axes[i + 1]
        ax.imshow(data['image'], cmap='gray')
        ctr = data['ctr']
        ctr_str = f"CTR: {ctr:.3f}" if ctr else "CTR: N/A"
        status = "✓ Healthy" if ctr and ctr < 0.5 else "○ Still enlarged"
        ax.set_title(f"Epoch {epoch}\n{ctr_str} {status}", fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f"Checkpoint Comparison: {test_image_path.name}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save comparison
    comparison_path = output_dir / f"{test_image_path.stem}_checkpoint_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Comparison saved: {comparison_path}")
    
    # Save individual results
    for epoch, data in results.items():
        result_path = output_dir / f"{test_image_path.stem}_epoch_{epoch}.png"
        cv2.imwrite(str(result_path), data['image_full'])
    print(f"  ✓ Individual results saved to: {output_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Epoch':<10} {'CTR':<10} {'Status':<20}")
    print("-" * 40)
    print(f"{'Original':<10} {original_ctr:.3f if original_ctr else 'N/A':<10} {'Cardiomegaly':<20}")
    
    for epoch, data in sorted(results.items()):
        ctr = data['ctr']
        if ctr:
            status = "✓ Healthy" if ctr < 0.5 else "Still enlarged"
            print(f"{epoch:<10} {ctr:.3f:<10} {status:<20}")
        else:
            print(f"{epoch:<10} {'N/A':<10} {'Error':<20}")
    
    # Find best epoch
    valid_results = {e: d for e, d in results.items() if d['ctr'] is not None}
    if valid_results:
        best_epoch = min(valid_results.keys(), key=lambda e: valid_results[e]['ctr'])
        best_ctr = valid_results[best_epoch]['ctr']
        print(f"\n🏆 Best result: Epoch {best_epoch} (CTR: {best_ctr:.3f})")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
