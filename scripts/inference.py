#!/usr/bin/env python
"""
Inference script for single image or batch processing.

Converts cardiomegaly chest X-rays to healthy versions.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, resolve_path


def _try_load_temperature_from_calibration_json(path: Path) -> Optional[float]:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        t = data.get("temperature", None)
        if t is None:
            return None
        t_f = float(t)
        if not (t_f > 0.0):
            return None
        return t_f
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on chest X-ray images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (file or directory)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to LoRA checkpoint (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--no-classifier",
        action="store_true",
        help="Skip classifier validation"
    )
    parser.add_argument(
        "--classifier-calibration",
        type=str,
        default="outputs/classifier/dataset_a_calibration.json",
        help=(
            "Path to classifier calibration JSON (written by scripts/calibrate_classifier_temperature.py). "
            "Used only if --classifier-temperature is not set. Ignored if missing."
        ),
    )
    parser.add_argument(
        "--disable-classifier-calibration",
        action="store_true",
        help="Disable loading temperature from calibration JSON (falls back to config/default)",
    )
    parser.add_argument(
        "--classifier-temperature",
        type=float,
        default=None,
        help=(
            "Softmax temperature for classifier scores (T>1 reduces overconfidence; "
            "use a value fitted on a validation set). If omitted, defaults to 1.0."
        ),
    )
    parser.add_argument(
        "--no-anatomical",
        action="store_true",
        help="Skip anatomical validation"
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save side-by-side comparison"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=5,
        help="Number of candidates to generate"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="For batch input dirs: maximum number of images to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    project_root = Path(__file__).parent.parent

    classifier_temperature: float = 1.0
    classifier_temperature_source: str = "default"
    if args.classifier_temperature is not None:
        classifier_temperature = float(args.classifier_temperature)
        classifier_temperature_source = "cli"
    elif (not args.disable_classifier_calibration) and args.classifier_calibration:
        calibration_path = resolve_path(args.classifier_calibration, project_root)
        t = _try_load_temperature_from_calibration_json(calibration_path)
        if t is not None:
            classifier_temperature = float(t)
            classifier_temperature_source = f"calibration_json:{calibration_path}"
        else:
            # Optional config support if present
            try:
                classifier_temperature = float(config.models.get('classifier_temperature', 1.0))
                classifier_temperature_source = "config"
            except Exception:
                classifier_temperature = 1.0
                classifier_temperature_source = "default"
    else:
        # Optional config support if present
        try:
            classifier_temperature = float(config.models.get('classifier_temperature', 1.0))
            classifier_temperature_source = "config"
        except Exception:
            classifier_temperature = 1.0
            classifier_temperature_source = "default"
    
    # Determine if input is file or directory
    input_path = Path(args.input)
    is_batch = input_path.is_dir()
    
    print("Inference Configuration")
    print("=" * 50)
    print(f"Input: {input_path}")
    print(f"Mode: {'Batch' if is_batch else 'Single'}")
    print(f"Device: {args.device}")
    if not args.no_classifier:
        print(f"Classifier temperature: T={classifier_temperature:.4g} (source: {classifier_temperature_source})")
    print("=" * 50)
    
    # Initialize pipeline
    from src.models.classifier import CardiomegalyClassifier
    from src.models.segmenter import ChexMaskSegmenter
    from src.models.inpainter import CardiacInpainter
    from src.validation.anatomical import AnatomicalValidator
    from src.inference.pipeline import CardiomegalyToHealthyPipeline
    from src.inference.batch_processor import BatchProcessor, process_single_image
    
    print("\nLoading models...")
    
    # Classifier (optional)
    classifier = None
    if not args.no_classifier:
        classifier_path = resolve_path(config.models.classifier_path, project_root)
        if classifier_path.exists():
            classifier = CardiomegalyClassifier(
                weights_path=classifier_path,
                device=args.device,
                temperature=float(classifier_temperature),
            )
            print("  ✓ Classifier loaded")
        else:
            print("  ⚠ Classifier not found, proceeding without")
    
    # Segmenter
    weights_dir = resolve_path(config.models.chexmask_weights_dir, project_root)
    if not weights_dir.exists():
        print(f"  ✗ Segmenter weights not found at {weights_dir}")
        sys.exit(1)
    
    segmenter = ChexMaskSegmenter(
        weights_dir=weights_dir,
        device=args.device
    )
    print("  ✓ Segmenter loaded")
    
    # Inpainter
    lora_path = args.checkpoint
    if not lora_path and hasattr(config.models, 'lora_weights_path'):
        lora_path = config.models.lora_weights_path

    resolved_lora_path = None
    if lora_path:
        if isinstance(lora_path, str) and ('<' in lora_path or '>' in lora_path):
            print("  ✗ Invalid --checkpoint value: angle brackets detected")
            print("    Don't use placeholders like models/inpainting/<checkpoint> in bash.")
            print("    Use a real folder, e.g.: --checkpoint models/inpainting/final_model")
            sys.exit(2)

        resolved_lora_path = resolve_path(lora_path, project_root)
        if args.checkpoint and not resolved_lora_path.exists():
            print(f"  ✗ Checkpoint path not found: {resolved_lora_path}")
            inpainting_root = project_root / "models" / "inpainting"
            if inpainting_root.exists():
                candidates = sorted([p.name for p in inpainting_root.iterdir() if p.is_dir()])
                if candidates:
                    print("    Available checkpoints under models/inpainting:")
                    for name in candidates:
                        print(f"      - {name}")
            sys.exit(1)
        if (not args.checkpoint) and resolved_lora_path and (not resolved_lora_path.exists()):
            print(f"  ⚠ LoRA weights path from config not found: {resolved_lora_path}")
            print("    Proceeding without LoRA (base inpainting model only).")
    
    inpainter = CardiacInpainter(
        base_model_id=config.models.base_inpainting_model,
        lora_weights_path=resolved_lora_path,
        device=args.device
    )
    print("  ✓ Inpainter loaded")
    
    # Validator (optional)
    validator = None
    if not args.no_anatomical:
        validator = AnatomicalValidator(
            segmenter=segmenter,
            min_ctr=config.validation.min_ctr,
            max_ctr=config.validation.max_ctr
        )
        print("  ✓ Validator initialized")
    
    # Create pipeline
    pipeline = CardiomegalyToHealthyPipeline(
        classifier=classifier,
        segmenter=segmenter,
        inpainter=inpainter,
        validator=validator,
        config=config
    )
    
    # Override settings from args
    pipeline.num_candidates = args.num_candidates
    if args.no_classifier:
        pipeline.require_classifier = False
    if args.no_anatomical:
        pipeline.require_anatomical = False
    
    print("\nProcessing...")
    
    if is_batch:
        # Batch processing
        output_dir = Path(args.output) if args.output else project_root / "outputs/generated"
        
        processor = BatchProcessor(
            pipeline=pipeline,
            output_dir=output_dir
        )
        
        summary = processor.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            save_comparisons=args.save_comparison,
            max_images=args.max_images
        )
        
        print(f"\nBatch processing complete!")
        print(f"Results saved to: {output_dir}")
        
    else:
        # Single image processing
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        
        output_path = args.output
        if not output_path:
            output_path = input_path.parent / f"{input_path.stem}_healthy.png"
        
        result = process_single_image(
            pipeline=pipeline,
            image_path=input_path,
            output_path=output_path,
            save_comparison=args.save_comparison
        )
        
        if result['success']:
            print(f"\n✓ Processing successful!")
            if result.get('output_ctr'):
                print(f"  CTR: {result['output_ctr']:.3f}")
            if result.get('output_confidence'):
                print(f"  Confidence: {result['output_confidence']:.2f}")
        else:
            print(f"\n✗ Processing failed")
            if result.get('error'):
                print(f"  Error: {result['error']}")


if __name__ == "__main__":
    main()
