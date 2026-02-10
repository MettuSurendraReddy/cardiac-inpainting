#!/usr/bin/env python
"""
Evaluation script for the cardiac inpainting model.

Evaluates trained model on test data and computes metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Evaluate the inpainting model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to evaluate"
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save generated images"
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
            try:
                classifier_temperature = float(config.models.get('classifier_temperature', 1.0))
                classifier_temperature_source = "config"
            except Exception:
                classifier_temperature = 1.0
                classifier_temperature_source = "default"
    else:
        try:
            classifier_temperature = float(config.models.get('classifier_temperature', 1.0))
            classifier_temperature_source = "config"
        except Exception:
            classifier_temperature = 1.0
            classifier_temperature_source = "default"
    
    # Setup paths
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Evaluation Configuration")
    print("=" * 50)
    print(f"Test directory: {test_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Classifier temperature: T={classifier_temperature:.4g} (source: {classifier_temperature_source})")
    print("=" * 50)
    
    # Check test directory exists
    if not test_dir.exists():
        print(f"\nError: Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Import modules
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    
    from src.models.classifier import CardiomegalyClassifier
    from src.models.segmenter import ChexMaskSegmenter
    from src.models.inpainter import CardiacInpainter
    from src.validation.anatomical import AnatomicalValidator
    from src.validation.metrics import EvaluationMetrics, calculate_ssim_masked
    from src.inference.pipeline import CardiomegalyToHealthyPipeline
    
    # Initialize components
    print("\nLoading models...")
    
    # Classifier
    classifier = None
    classifier_path = resolve_path(config.models.classifier_path, project_root)
    if classifier_path.exists():
        classifier = CardiomegalyClassifier(
            weights_path=classifier_path,
            device=args.device,
            temperature=float(classifier_temperature),
        )
        print("  ✓ Classifier loaded")
    else:
        print("  ✗ Classifier not found, skipping classifier validation")
    
    # Segmenter
    segmenter = None
    weights_dir = resolve_path(config.models.chexmask_weights_dir, project_root)
    if weights_dir.exists():
        segmenter = ChexMaskSegmenter(
            weights_dir=weights_dir,
            device=args.device
        )
        print("  ✓ Segmenter loaded")
    else:
        print("  ✗ Segmenter not found")
        sys.exit(1)
    
    # Inpainter
    inpainter = CardiacInpainter(
        base_model_id=config.models.base_inpainting_model,
        lora_weights_path=args.checkpoint,
        device=args.device
    )
    print("  ✓ Inpainter loaded")
    
    # Validator
    validator = AnatomicalValidator(
        segmenter=segmenter,
        min_ctr=config.validation.min_ctr,
        max_ctr=config.validation.max_ctr
    )
    
    # Pipeline
    pipeline = CardiomegalyToHealthyPipeline(
        classifier=classifier,
        segmenter=segmenter,
        inpainter=inpainter,
        validator=validator,
        config=config
    )
    
    # Find test images
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    test_images = [
        p for p in test_dir.iterdir()
        if p.suffix.lower() in valid_extensions
    ]
    
    if args.max_images:
        test_images = test_images[:args.max_images]
    
    print(f"\nFound {len(test_images)} test images")
    
    # Initialize metrics tracker
    metrics = EvaluationMetrics()
    
    # Process images
    results = []
    
    for img_path in tqdm(test_images, desc="Evaluating"):
        try:
            # Load original image
            original = Image.open(img_path).convert('L')
            
            # Get input CTR
            masks = segmenter.segment(original)
            input_ctr = validator.calculate_ctr(
                original,
                heart_mask=masks['heart'],
                lung_mask=masks['lungs']
            )
            
            # Process
            result = pipeline.process_with_details(
                img_path,
                skip_verification=True
            )
            
            if result['success'] and result['output'] is not None:
                generated = result['output']
                
                # Calculate SSIM on non-masked regions
                ssim_result = calculate_ssim_masked(
                    original,
                    generated,
                    masks['heart']
                )
                
                # Add to metrics
                metrics.add_result(
                    success=True,
                    ctr=result.get('output_ctr'),
                    classifier_pred='healthy' if classifier else None,
                    classifier_conf=result.get('output_confidence'),
                    ssim=ssim_result.get('outside'),
                    metadata={
                        'path': str(img_path),
                        'input_ctr': input_ctr
                    }
                )
                
                # Save output if requested
                if args.save_outputs:
                    out_path = output_dir / 'generated' / f"{img_path.stem}_healthy.png"
                    out_path.parent.mkdir(exist_ok=True)
                    generated.save(out_path)
                
                results.append({
                    'path': str(img_path),
                    'success': True,
                    'input_ctr': input_ctr,
                    'output_ctr': result.get('output_ctr'),
                    'confidence': result.get('output_confidence'),
                    'ssim': ssim_result.get('outside')
                })
            else:
                metrics.add_result(success=False)
                results.append({
                    'path': str(img_path),
                    'success': False,
                    'input_ctr': input_ctr
                })
                
        except Exception as e:
            metrics.add_result(success=False)
            results.append({
                'path': str(img_path),
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    metrics.print_summary()
    
    # Save detailed results
    summary = metrics.get_summary()
    summary['checkpoint'] = args.checkpoint
    summary['test_dir'] = str(test_dir)
    summary['timestamp'] = datetime.now().isoformat()
    summary['classifier_temperature'] = float(classifier_temperature)
    summary['classifier_temperature_source'] = classifier_temperature_source
    summary['results'] = results
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    # Check success criteria
    print("\n" + "=" * 50)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 50)
    
    criteria_met = True
    
    # 80%+ success rate
    if summary['success_rate'] >= 0.80:
        print(f"✓ Success rate: {summary['success_rate']:.1%} >= 80%")
    else:
        print(f"✗ Success rate: {summary['success_rate']:.1%} < 80%")
        criteria_met = False
    
    # CTR in healthy range
    if summary.get('ctr'):
        ctr_mean = summary['ctr']['mean']
        if 0.35 <= ctr_mean <= 0.50:
            print(f"✓ Mean CTR: {ctr_mean:.3f} in [0.35, 0.50]")
        else:
            print(f"✗ Mean CTR: {ctr_mean:.3f} not in [0.35, 0.50]")
            criteria_met = False
    
    # Classifier accuracy
    if summary.get('classifier'):
        healthy_rate = summary['classifier']['healthy_rate']
        if healthy_rate >= 0.80:
            print(f"✓ Healthy classification rate: {healthy_rate:.1%} >= 80%")
        else:
            print(f"✗ Healthy classification rate: {healthy_rate:.1%} < 80%")
            criteria_met = False
    
    # SSIM on preserved regions
    if summary.get('ssim'):
        ssim_mean = summary['ssim']['mean']
        if ssim_mean >= 0.95:
            print(f"✓ Mean SSIM (preserved): {ssim_mean:.4f} >= 0.95")
        else:
            print(f"✗ Mean SSIM (preserved): {ssim_mean:.4f} < 0.95")
            criteria_met = False
    
    print("=" * 50)
    if criteria_met:
        print("ALL SUCCESS CRITERIA MET! ✓")
    else:
        print("Some criteria not met. Review results for improvement areas.")
    print("=" * 50)


if __name__ == "__main__":
    main()
