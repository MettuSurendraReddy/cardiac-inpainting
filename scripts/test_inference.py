#!/usr/bin/env python
"""
Quick test script for the cardiac inpainting pipeline.

Tests the full pipeline on a single cardiomegaly image.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "CheXmask-Database" / "HybridGNet"))


def find_latest_checkpoint(models_dir: Path) -> Path:
    """Find the latest checkpoint in the models directory."""
    inpainting_dir = models_dir / "inpainting"
    
    if not inpainting_dir.exists():
        raise FileNotFoundError(f"No inpainting directory found at {inpainting_dir}")
    
    # Look for checkpoint_epoch_* directories
    checkpoints = []
    for path in inpainting_dir.iterdir():
        if path.is_dir() and path.name.startswith('checkpoint_epoch_'):
            try:
                epoch_num = int(path.name.split('_')[-1])
                checkpoints.append((epoch_num, path))
            except ValueError:
                continue
    
    if not checkpoints:
        # Try best_model as fallback
        best_model = inpainting_dir / "best_model"
        if best_model.exists():
            return best_model
        raise FileNotFoundError("No checkpoints found")
    
    # Return the checkpoint with highest epoch number
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def main():
    parser = argparse.ArgumentParser(description="Test cardiac inpainting pipeline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint (default: latest)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image (default: first cardiomegaly image)"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidates to generate"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cardiac Inpainting Pipeline Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Paths
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_latest_checkpoint(PROJECT_ROOT / "models")
        print(f"\nUsing latest checkpoint: {checkpoint_path.name}")
    
    test_image_dir = PROJECT_ROOT / "data" / "raw" / "cardiomegaly"
    output_dir = PROJECT_ROOT / "outputs" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a test image
    if args.image:
        test_image_path = Path(args.image)
    else:
        test_images = list(test_image_dir.glob("*.png"))
        if not test_images:
            print("No test images found in data/raw/cardiomegaly/")
            return
        test_image_path = test_images[0]
    
    print(f"\nTest image: {test_image_path.name}")
    
    # Step 1: Load segmenter and generate mask
    print("\n[1/4] Loading segmenter...")
    from scripts.generate_masks import load_hybridgnet, process_image
    
    weights_path = PROJECT_ROOT / "models" / "CheXmask-Database" / "Weights" / "SegmentationModel" / "bestMSE.pt"
    segmenter = load_hybridgnet(weights_path, device)
    print("  ✓ Segmenter loaded")
    
    # Generate mask
    print("\n[2/4] Generating heart mask...")
    masks = process_image(segmenter, test_image_path, device)
    heart_mask = masks['heart']
    print(f"  ✓ Mask generated (shape: {heart_mask.shape})")
    
    # Step 3: Load inpainting model
    print("\n[3/4] Loading inpainting model...")
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel, LoraConfig, get_peft_model
    
    # Load base model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights
    model_pt = checkpoint_path / "model.pt"
    lora_dir = checkpoint_path / "lora_weights"
    
    if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
        print(f"  Loading LoRA from PEFT format: {lora_dir}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir)
    elif model_pt.exists():
        print(f"  Loading LoRA from state_dict: {model_pt}")
        # Recreate LoRA config
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
        print("  ⚠ No LoRA weights found, using base model")
    
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    print("  ✓ Inpainting model loaded")
    
    # Step 4: Run inpainting
    print("\n[4/4] Running inpainting...")
    
    # Load and preprocess image
    import cv2
    original_img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    original_size = original_img.shape[:2]
    
    # Resize to 512x512 for SD
    img_resized = cv2.resize(original_img, (512, 512))
    mask_resized = cv2.resize(heart_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Convert to PIL (RGB for SD)
    img_pil = Image.fromarray(cv2.cvtColor(
        np.stack([img_resized]*3, axis=-1), 
        cv2.COLOR_BGR2RGB
    ))
    
    # Dilate mask slightly for better coverage
    kernel = np.ones((15, 15), np.uint8)
    mask_dilated = cv2.dilate(mask_resized, kernel, iterations=2)
    mask_pil = Image.fromarray(mask_dilated)
    
    # Run inpainting - CONDITIONAL with prompts from project.md
    prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
    negative_prompt = "enlarged heart, cardiomegaly, artifacts, blur, noise"
    
    num_candidates = args.num_candidates
    print(f"  Generating {num_candidates} candidates...")
    candidates = []
    for i in range(num_candidates):
        print(f"    Candidate {i+1}/{num_candidates}...", end=" ")
        with torch.autocast(device):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                num_inference_steps=50,
                guidance_scale=7.5,  # From project.md
                generator=torch.Generator(device).manual_seed(42 + i)
            ).images[0]
        candidates.append(result)
        print("✓")
    
    # Use the first candidate for now
    result = candidates[0]
    print("  ✓ Inpainting complete")
    
    # Convert result to grayscale
    result_gray = np.array(result.convert('L'))
    
    # Resize back to original size
    result_final = cv2.resize(result_gray, (original_size[1], original_size[0]))
    
    # Save results
    output_path = output_dir / f"{test_image_path.stem}_result.png"
    cv2.imwrite(str(output_path), result_final)
    print(f"\n✓ Result saved to: {output_path}")
    
    # Save all candidates
    for i, cand in enumerate(candidates):
        cand_gray = np.array(cand.convert('L'))
        cand_final = cv2.resize(cand_gray, (original_size[1], original_size[0]))
        cand_path = output_dir / f"{test_image_path.stem}_candidate_{i+1}.png"
        cv2.imwrite(str(cand_path), cand_final)
    print(f"✓ All {len(candidates)} candidates saved")
    
    # Create comparison visualization with all candidates
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title('Original (Cardiomegaly)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_resized, cmap='gray')
    axes[0, 1].set_title('Heart Mask')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mask_dilated, cmap='gray')
    axes[0, 2].set_title('Dilated Mask (Inpaint Area)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(result_gray, cmap='gray')
    axes[0, 3].set_title('Best Result')
    axes[0, 3].axis('off')
    
    # Show first 4 candidates
    for i in range(min(4, len(candidates))):
        cand_gray = np.array(candidates[i].convert('L'))
        axes[1, i].imshow(cand_gray, cmap='gray')
        axes[1, i].set_title(f'Candidate {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    comparison_path = output_dir / f"{test_image_path.stem}_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison saved to: {comparison_path}")
    
    # Calculate CTR for validation
    print("\n" + "=" * 60)
    print("Validating result...")
    
    # Get masks for result
    result_512 = cv2.resize(result_final, (512, 512))
    cv2.imwrite(str(output_dir / "temp_result.png"), result_512)
    
    result_masks = process_image(segmenter, output_dir / "temp_result.png", device)
    
    # Calculate CTR
    from src.validation.anatomical import AnatomicalValidator
    
    def calculate_ctr(heart_mask, lung_mask):
        """Calculate cardiothoracic ratio."""
        # Find heart bounding box
        heart_coords = np.where(heart_mask > 0)
        if len(heart_coords[0]) == 0:
            return None
        heart_width = heart_coords[1].max() - heart_coords[1].min()
        
        # Find lung bounding box (chest width)
        lung_coords = np.where(lung_mask > 0)
        if len(lung_coords[0]) == 0:
            return None
        chest_width = lung_coords[1].max() - lung_coords[1].min()
        
        if chest_width == 0:
            return None
        
        return heart_width / chest_width
    
    original_masks = process_image(segmenter, test_image_path, device)
    
    original_ctr = calculate_ctr(original_masks['heart'], original_masks['lungs'])
    result_ctr = calculate_ctr(result_masks['heart'], result_masks['lungs'])
    
    print(f"\nOriginal CTR: {original_ctr:.3f}" if original_ctr else "Original CTR: N/A")
    print(f"Result CTR:   {result_ctr:.3f}" if result_ctr else "Result CTR: N/A")
    
    if original_ctr and result_ctr:
        if result_ctr < 0.5:
            print(f"\n✓ SUCCESS: CTR reduced from {original_ctr:.3f} to {result_ctr:.3f} (< 0.5 = healthy)")
        else:
            print(f"\n⚠ CTR still {result_ctr:.3f} >= 0.5 (try more training or different seed)")
    
    # Cleanup
    (output_dir / "temp_result.png").unlink(missing_ok=True)
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
