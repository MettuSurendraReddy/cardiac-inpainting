"""
Batch processing utilities for the inference pipeline.

Provides efficient processing of multiple images with progress tracking.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import numpy as np
from PIL import Image
from tqdm import tqdm


class BatchProcessor:
    """
    Batch processor for running the inpainting pipeline on multiple images.
    
    Features:
    - Progress tracking
    - Parallel processing (where applicable)
    - Automatic result saving
    - Statistics collection
    """
    
    def __init__(
        self,
        pipeline,
        num_workers: int = 1,  # GPU inference is usually single-threaded
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            pipeline: CardiomegalyToHealthyPipeline instance
            num_workers: Number of parallel workers (usually 1 for GPU)
            save_results: Whether to save results automatically
            output_dir: Directory for saving results
        """
        self.pipeline = pipeline
        self.num_workers = num_workers
        self.save_results = save_results
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/generated")
        
        # Create output directory
        if save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_comparisons: bool = True,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
        max_images: Optional[int] = None
    ) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory (overrides default)
            save_comparisons: Save side-by-side comparisons
            extensions: Valid file extensions
            max_images: Maximum number of images to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(set(image_paths))
        
        if max_images:
            image_paths = image_paths[:max_images]
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = []
        success_count = 0
        ctrs = []
        confidences = []
        
        for img_path in tqdm(image_paths, desc="Processing"):
            try:
                result = self._process_single(
                    img_path,
                    output_dir,
                    save_comparisons
                )
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    if result.get('output_ctr'):
                        ctrs.append(result['output_ctr'])
                    if result.get('output_confidence'):
                        confidences.append(result['output_confidence'])
                        
            except Exception as e:
                results.append({
                    'input_path': str(img_path),
                    'success': False,
                    'error': str(e)
                })
        
        # Compute statistics
        stats = {
            'total': len(image_paths),
            'successful': success_count,
            'failed': len(image_paths) - success_count,
            'success_rate': success_count / max(1, len(image_paths))
        }
        
        if ctrs:
            stats['ctr'] = {
                'mean': float(np.mean(ctrs)),
                'std': float(np.std(ctrs)),
                'min': float(np.min(ctrs)),
                'max': float(np.max(ctrs))
            }
        
        if confidences:
            stats['confidence'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            }
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': str(input_dir),
            'output_dir': str(output_dir),
            'statistics': stats,
            'results': results
        }
        
        summary_path = output_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nProcessing complete!")
        print(f"Success rate: {stats['success_rate']:.1%}")
        if ctrs:
            print(f"Average CTR: {stats['ctr']['mean']:.3f}")
        
        return summary
    
    def _process_single(
        self,
        image_path: Path,
        output_dir: Path,
        save_comparison: bool = True
    ) -> Dict:
        """Process a single image."""
        # Process with pipeline
        result = self.pipeline.process_with_details(image_path)
        
        if result['success'] and result['output'] is not None:
            # Save generated image
            output_name = f"{image_path.stem}_healthy.png"
            output_path = output_dir / output_name
            result['output'].save(output_path)
            result['output_path'] = str(output_path)
            
            # Save comparison if requested
            if save_comparison:
                comparison_dir = output_dir / 'comparisons'
                comparison_dir.mkdir(exist_ok=True)
                
                comparison = self._create_comparison(
                    Image.open(image_path).convert('L'),
                    result['output']
                )
                comparison_path = comparison_dir / f"{image_path.stem}_comparison.png"
                comparison.save(comparison_path)
                result['comparison_path'] = str(comparison_path)
        
        result['input_path'] = str(image_path)
        
        # Remove PIL image from result (not JSON serializable)
        if 'output' in result and result['output'] is not None:
            result['output'] = str(result.get('output_path', 'saved'))
        if 'candidates' in result:
            result['candidates'] = len(result['candidates'])
        
        return result
    
    def _create_comparison(
        self,
        original: Image.Image,
        generated: Image.Image,
        padding: int = 10
    ) -> Image.Image:
        """
        Create a side-by-side comparison image.
        
        Args:
            original: Original image
            generated: Generated image
            padding: Padding between images
            
        Returns:
            Combined comparison image
        """
        # Ensure same size
        if original.size != generated.size:
            generated = generated.resize(original.size, Image.Resampling.LANCZOS)
        
        # Create comparison
        width = original.width * 2 + padding
        height = original.height
        
        comparison = Image.new('L', (width, height), color=128)
        comparison.paste(original, (0, 0))
        comparison.paste(generated, (original.width + padding, 0))
        
        return comparison
    
    def process_list(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        save_comparisons: bool = True
    ) -> Dict:
        """
        Process a list of image paths.
        
        Args:
            image_paths: List of paths to process
            output_dir: Output directory
            save_comparisons: Save side-by-side comparisons
            
        Returns:
            Dictionary with processing results
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        success_count = 0
        
        for img_path in tqdm(image_paths, desc="Processing"):
            try:
                result = self._process_single(
                    Path(img_path),
                    output_dir,
                    save_comparisons
                )
                results.append(result)
                
                if result['success']:
                    success_count += 1
                    
            except Exception as e:
                results.append({
                    'input_path': str(img_path),
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'total': len(image_paths),
            'successful': success_count,
            'failed': len(image_paths) - success_count,
            'success_rate': success_count / max(1, len(image_paths)),
            'results': results
        }


def process_single_image(
    pipeline,
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    save_comparison: bool = True
) -> Dict:
    """
    Convenience function to process a single image.
    
    Args:
        pipeline: CardiomegalyToHealthyPipeline instance
        image_path: Path to input image
        output_path: Path for output image
        save_comparison: Save side-by-side comparison
        
    Returns:
        Processing result dictionary
    """
    image_path = Path(image_path)
    
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_healthy.png"
    output_path = Path(output_path)
    
    # Process
    result = pipeline.process_with_details(image_path)
    
    if result['success'] and result['output'] is not None:
        # Save output
        result['output'].save(output_path)
        print(f"Saved: {output_path}")
        
        if save_comparison:
            # Create and save comparison
            original = Image.open(image_path).convert('L')
            generated = result['output']
            
            if original.size != generated.size:
                generated = generated.resize(original.size)
            
            width = original.width * 2 + 10
            comparison = Image.new('L', (width, original.height), 128)
            comparison.paste(original, (0, 0))
            comparison.paste(generated, (original.width + 10, 0))
            
            comparison_path = output_path.parent / f"{image_path.stem}_comparison.png"
            comparison.save(comparison_path)
            print(f"Saved comparison: {comparison_path}")
    else:
        print(f"Processing failed for {image_path}")
    
    return result


def create_results_report(
    results: List[Dict],
    output_path: Union[str, Path]
) -> None:
    """
    Create an HTML report of processing results.
    
    Args:
        results: List of result dictionaries
        output_path: Path for the HTML report
    """
    output_path = Path(output_path)
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Processing Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .success { color: green; }
            .failed { color: red; }
            img { max-width: 200px; }
        </style>
    </head>
    <body>
        <h1>Cardiac Inpainting Results</h1>
        <table>
            <tr>
                <th>Input</th>
                <th>Status</th>
                <th>Input CTR</th>
                <th>Output CTR</th>
                <th>Confidence</th>
                <th>Comparison</th>
            </tr>
    """
    
    for r in results:
        status = 'success' if r.get('success') else 'failed'
        status_class = status
        
        input_ctr = f"{r.get('input_ctr', 'N/A'):.3f}" if r.get('input_ctr') else 'N/A'
        output_ctr = f"{r.get('output_ctr', 'N/A'):.3f}" if r.get('output_ctr') else 'N/A'
        confidence = f"{r.get('output_confidence', 'N/A'):.2f}" if r.get('output_confidence') else 'N/A'
        
        comparison_img = ''
        if r.get('comparison_path'):
            comparison_img = f'<img src="{r["comparison_path"]}">'
        
        html += f"""
            <tr>
                <td>{Path(r.get('input_path', 'Unknown')).name}</td>
                <td class="{status_class}">{status}</td>
                <td>{input_ctr}</td>
                <td>{output_ctr}</td>
                <td>{confidence}</td>
                <td>{comparison_img}</td>
            </tr>
        """
    
    html += """
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")
