#!/usr/bin/env python
"""
Generate segmentation masks using CheXMask HybridGNet.

Generates heart and lung masks for all images in a directory.
Uses the HybridGNet model directly from the CheXmask-Database.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import scipy.sparse as sp

# Add project root and HybridGNet to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "CheXmask-Database" / "HybridGNet"))

from src.config import load_config


def get_dense_mask(landmarks, h=1024, w=1024):
    """Convert landmarks to binary masks."""
    rl = landmarks[:44].reshape(-1, 1, 2).astype(np.int32)
    ll = landmarks[44:94].reshape(-1, 1, 2).astype(np.int32)
    heart = landmarks[94:].reshape(-1, 1, 2).astype(np.int32)
    
    # Create separate masks
    right_lung_mask = np.zeros([h, w], dtype=np.uint8)
    left_lung_mask = np.zeros([h, w], dtype=np.uint8)
    heart_mask = np.zeros([h, w], dtype=np.uint8)
    
    cv2.drawContours(right_lung_mask, [rl], -1, 255, -1)
    cv2.drawContours(left_lung_mask, [ll], -1, 255, -1)
    cv2.drawContours(heart_mask, [heart], -1, 255, -1)
    
    return heart_mask, right_lung_mask, left_lung_mask


def load_hybridgnet(weights_path: Path, device: str):
    """Load the HybridGNet model."""
    from models.HybridGNet2IGSC import Hybrid
    from utils.utils import scipy_to_torch_sparse, genMatrixesLungsHeart
    
    # Generate graph matrices
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
    config = {
        'n_nodes': [N1, N1, N1, N2, N2, N2],
        'latents': 64,
        'inputsize': 1024,
        'filters': [2, 32, 32, 32, 16, 16, 16],
        'skip_features': 32
    }
    
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = (
        [scipy_to_torch_sparse(x).to(device) for x in X]
        for X in (A_, D_, U_)
    )
    
    # Create and load model
    model = Hybrid(config.copy(), D_t, U_t, A_t).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model


def process_image(model, image_path: Path, device: str):
    """Process a single image and return masks."""
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_h, original_w = img.shape
    
    # Resize to 1024x1024 for model
    img_resized = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Create input tensor
    input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(device).float()
    
    # Get landmarks
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, (list, tuple)) and len(output) > 1:
            output = output[0]
    
    # Convert to coordinates
    landmarks = output.cpu().numpy().reshape(-1, 2) * 1024
    landmarks = landmarks.round().astype(np.int32)
    
    # Get masks at 1024x1024
    heart_mask, right_lung_mask, left_lung_mask = get_dense_mask(landmarks)
    
    # Resize masks to original image size
    heart_mask = cv2.resize(heart_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    right_lung_mask = cv2.resize(right_lung_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    left_lung_mask = cv2.resize(left_lung_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    return {
        'heart': heart_mask,
        'right_lung': right_lung_mask,
        'left_lung': left_lung_mask,
        'lungs': np.maximum(right_lung_mask, left_lung_mask),
        'landmarks': landmarks
    }


def main():
    parser = argparse.ArgumentParser(description="Generate segmentation masks using CheXMask HybridGNet")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing input images (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output masks (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["healthy", "cardiomegaly", "both"],
        default="both",
        help="Which category to process"
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        choices=["heart", "lungs", "all"],
        default="heart",
        help="Type of mask to save"
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="Save visualization of masks overlaid on images"
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process at most N images per category (useful for quick debugging)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback on errors"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup paths from config or arguments
    raw_dir = Path(args.input_dir) if args.input_dir else PROJECT_ROOT / "data" / "raw"
    masks_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "masks"
    weights_path = PROJECT_ROOT / "models" / "CheXmask-Database" / "Weights" / "SegmentationModel" / "bestMSE.pt"
    
    # Determine which categories to process
    if args.category == "both":
        categories = ["healthy", "cardiomegaly"]
    else:
        categories = [args.category]
    
    print("=" * 60)
    print("CheXMask Heart Segmentation - Mask Generation")
    print("=" * 60)
    print(f"Raw images directory: {raw_dir}")
    print(f"Output masks directory: {masks_dir}")
    print(f"Model weights: {weights_path}")
    print(f"Device: {args.device}")
    print(f"Categories: {categories}")
    print(f"Mask type: {args.mask_type}")
    print("=" * 60)
    
    # Check weights exist
    if not weights_path.exists():
        print(f"\n❌ Error: Weights not found at {weights_path}")
        print("Please ensure CheXMask weights are downloaded.")
        sys.exit(1)
    
    # Load model
    print("\n📦 Loading HybridGNet model...")
    try:
        model = load_hybridgnet(weights_path, args.device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process each category
    total_success = 0
    total_images = 0
    
    total_errors = 0

    for category in categories:
        input_dir = raw_dir / category
        output_dir = masks_dir / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not input_dir.exists():
            print(f"\n⚠️ Input directory not found: {input_dir}")
            continue
        
        # Find all images
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = [
            p for p in input_dir.iterdir()
            if p.suffix.lower() in valid_extensions
        ]

        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]
        
        print(f"\n📁 Processing {category}: {len(image_paths)} images")
        total_images += len(image_paths)
        
        # Create visualization directory if needed
        if args.save_visualization:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
        
        # Process images with progress bar
        from tqdm import tqdm
        for img_path in tqdm(image_paths, desc=f"  {category}"):
            try:
                # Process image
                masks = process_image(model, img_path, args.device)
                
                # Select which mask to save
                if args.mask_type == "heart":
                    mask = masks['heart']
                elif args.mask_type == "lungs":
                    mask = masks['lungs']
                else:  # all
                    # Combined: lungs=128, heart=255
                    mask = np.zeros_like(masks['heart'], dtype=np.uint8)
                    mask[masks['lungs'] > 0] = 128
                    mask[masks['heart'] > 0] = 255
                
                # Save mask
                mask_path = output_dir / f"{img_path.stem}.png"
                cv2.imwrite(str(mask_path), mask)
                
                # Save visualization if requested
                if args.save_visualization:
                    vis = create_visualization(img_path, masks)
                    vis_path = vis_dir / f"{img_path.stem}_vis.png"
                    cv2.imwrite(str(vis_path), vis)
                
                total_success += 1
                
            except Exception as e:
                total_errors += 1
                print(f"\n⚠️ Error processing {img_path.name}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Mask generation complete!")
    print(f"   Processed: {total_success}/{total_images} images")
    if total_errors:
        print(f"   Errors: {total_errors}")
    print(f"   Output: {masks_dir}")
    print("=" * 60)


def create_visualization(image_path: Path, masks: dict) -> np.ndarray:
    """Create a visualization with masks overlaid on image."""
    # Load original image
    img = cv2.imread(str(image_path))
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize masks if needed
    h, w = img.shape[:2]
    heart = masks['heart']
    lungs = masks['lungs']
    
    if heart.shape[:2] != (h, w):
        heart = cv2.resize(heart, (w, h), interpolation=cv2.INTER_NEAREST)
        lungs = cv2.resize(lungs, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay
    overlay = img.copy()
    
    # Lungs in blue (BGR)
    lung_mask = lungs > 0
    overlay[lung_mask, 0] = np.clip(overlay[lung_mask, 0].astype(np.int32) + 100, 0, 255).astype(np.uint8)
    
    # Heart in red (BGR)
    heart_mask = heart > 0
    overlay[heart_mask, 2] = np.clip(overlay[heart_mask, 2].astype(np.int32) + 100, 0, 255).astype(np.uint8)
    
    # Blend with original
    result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    
    return result


if __name__ == "__main__":
    main()
