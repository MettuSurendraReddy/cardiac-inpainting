"""
Flask API for Cardiac Inpainting Pipeline.

Provides endpoints for:
- Inpainting cardiomegaly images to healthy
- Evaluating model performance
- Listing available checkpoints
"""

import os
import sys
import io
import base64
import tempfile
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image
import cv2
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "CheXmask-Database" / "HybridGNet"))

app = Flask(__name__)
CORS(app)

# Global model cache
_models = {
    'segmenter': None,
    'inpainting_pipes': {},  # Cache by epoch
    'device': None
}


def get_device():
    """Get the compute device."""
    if _models['device'] is None:
        _models['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    return _models['device']


def get_segmenter():
    """Load and cache the segmentation model."""
    if _models['segmenter'] is None:
        from scripts.generate_masks import load_hybridgnet
        weights_path = PROJECT_ROOT / "models" / "CheXmask-Database" / "Weights" / "SegmentationModel" / "bestMSE.pt"
        _models['segmenter'] = load_hybridgnet(weights_path, get_device())
    return _models['segmenter']


def get_inpainting_pipe(epoch: int = None):
    """Load and cache inpainting pipeline for specific epoch."""
    from diffusers import StableDiffusionInpaintPipeline
    from peft import PeftModel, LoraConfig, get_peft_model
    
    device = get_device()
    
    # Determine checkpoint path
    if epoch is None:
        checkpoint_path = PROJECT_ROOT / "models" / "inpainting" / "best_model"
    else:
        checkpoint_path = PROJECT_ROOT / "models" / "inpainting" / f"checkpoint_epoch_{epoch}"
    
    cache_key = str(checkpoint_path)
    
    if cache_key not in _models['inpainting_pipes']:
        if not checkpoint_path.exists():
            return None
        
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
        _models['inpainting_pipes'][cache_key] = pipe
    
    return _models['inpainting_pipes'][cache_key]


def generate_mask(image_array, segmenter, device):
    """Generate heart mask for an image."""
    from scripts.generate_masks import process_image
    
    # Save temp file for process_image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, image_array)
    
    try:
        masks = process_image(segmenter, Path(temp_path), device)
        return masks
    finally:
        os.unlink(temp_path)


def calculate_ctr_from_masks(masks):
    """Calculate CTR from segmentation masks."""
    heart_mask = masks['heart']
    left_lung = masks['left_lung']
    right_lung = masks['right_lung']
    
    # Heart width
    heart_cols = np.where(heart_mask.sum(axis=0) > 0)[0]
    if len(heart_cols) == 0:
        return None
    heart_width = heart_cols[-1] - heart_cols[0]
    
    # Chest width
    combined_lungs = np.maximum(left_lung, right_lung)
    lung_cols = np.where(combined_lungs.sum(axis=0) > 0)[0]
    if len(lung_cols) == 0:
        return None
    chest_width = lung_cols[-1] - lung_cols[0]
    
    if chest_width == 0:
        return None
    
    return heart_width / chest_width


def image_to_base64(image_array):
    """Convert numpy array to base64 string."""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_string):
    """Convert base64 string to numpy array."""
    image_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(image_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)


# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'device': get_device(),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """List all available checkpoints."""
    checkpoints_dir = PROJECT_ROOT / "models" / "inpainting"
    
    if not checkpoints_dir.exists():
        return jsonify({'checkpoints': [], 'error': 'No checkpoints directory found'})
    
    checkpoints = []
    
    for path in checkpoints_dir.iterdir():
        if path.is_dir():
            if path.name.startswith('checkpoint_epoch_'):
                try:
                    epoch = int(path.name.split('_')[-1])
                    checkpoints.append({
                        'name': path.name,
                        'epoch': epoch,
                        'path': str(path)
                    })
                except ValueError:
                    pass
            elif path.name in ['best_model', 'final_model']:
                checkpoints.append({
                    'name': path.name,
                    'epoch': None,
                    'path': str(path)
                })
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x['epoch'] if x['epoch'] else 0)
    
    return jsonify({
        'checkpoints': checkpoints,
        'total': len(checkpoints)
    })


@app.route('/inpaint', methods=['POST'])
def inpaint():
    """
    Inpaint a cardiomegaly image to generate a healthy version.
    
    Request (JSON):
        - image: base64 encoded image OR
        - file: multipart file upload
        - epoch: (optional) checkpoint epoch to use
        - num_candidates: (optional) number of candidates to generate (default: 1)
        - return_format: (optional) 'base64' or 'file' (default: 'base64')
    
    Response:
        - result_image: base64 encoded result
        - original_ctr: CTR of input image
        - result_ctr: CTR of output image
        - success: whether CTR was reduced below 0.5
        - mask: base64 encoded mask used
    """
    device = get_device()
    
    # Get epoch parameter
    epoch = request.form.get('epoch') or request.json.get('epoch') if request.is_json else None
    if epoch is not None:
        epoch = int(epoch)
    
    num_candidates = int(request.form.get('num_candidates', 1) or 
                        (request.json.get('num_candidates', 1) if request.is_json else 1))
    
    return_format = request.form.get('return_format', 'base64') or \
                   (request.json.get('return_format', 'base64') if request.is_json else 'base64')
    
    # Get image
    if 'file' in request.files:
        file = request.files['file']
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif request.is_json and 'image' in request.json:
        original_img = base64_to_image(request.json['image'])
    else:
        return jsonify({'error': 'No image provided. Use "file" or "image" (base64)'}), 400
    
    if original_img is None:
        return jsonify({'error': 'Could not decode image'}), 400
    
    original_size = original_img.shape[:2]
    
    try:
        # Load models
        segmenter = get_segmenter()
        pipe = get_inpainting_pipe(epoch)
        
        if pipe is None:
            available = list_checkpoints().json['checkpoints']
            return jsonify({
                'error': f'Checkpoint epoch {epoch} not found',
                'available_checkpoints': available
            }), 404
        
        # Generate mask
        masks = generate_mask(original_img, segmenter, device)
        heart_mask = masks['heart']
        
        # Calculate original CTR
        original_ctr = calculate_ctr_from_masks(masks)
        
        # Prepare for inpainting
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
        
        # Run inpainting
        prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
        negative_prompt = "enlarged heart, cardiomegaly, artifacts, blur, noise"
        
        best_result = None
        best_ctr = 1.0
        
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
            
            # Convert and resize
            result_gray = np.array(result.convert('L'))
            result_full = cv2.resize(result_gray, (original_size[1], original_size[0]))
            
            # Calculate CTR
            result_masks = generate_mask(result_full, segmenter, device)
            result_ctr = calculate_ctr_from_masks(result_masks)
            
            if result_ctr and result_ctr < best_ctr:
                best_ctr = result_ctr
                best_result = result_full
        
        if best_result is None:
            best_result = result_full
            best_ctr = result_ctr
        
        # Prepare response - ensure Python native types for JSON serialization
        response = {
            'success': bool(best_ctr is not None and best_ctr < 0.5),
            'original_ctr': float(round(original_ctr, 4)) if original_ctr else None,
            'result_ctr': float(round(best_ctr, 4)) if best_ctr else None,
            'ctr_reduction': float(round(original_ctr - best_ctr, 4)) if original_ctr and best_ctr else None,
            'epoch_used': epoch,
            'num_candidates': num_candidates
        }
        
        if return_format == 'base64':
            response['result_image'] = image_to_base64(best_result)
            response['mask'] = image_to_base64(heart_mask)
            response['mask_dilated'] = image_to_base64(mask_dilated)
            return jsonify(response)
        else:
            # Return as file
            _, buffer = cv2.imencode('.png', best_result)
            return send_file(
                io.BytesIO(buffer),
                mimetype='image/png',
                as_attachment=True,
                download_name='inpainted_result.png'
            )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate model on an image with detailed statistics.
    
    Request (JSON):
        - image: base64 encoded image OR
        - file: multipart file upload
        - epochs: (optional) list of epochs to compare (default: all available)
    
    Response:
        - original: original image statistics
        - results: list of results per epoch
        - best_epoch: epoch with best CTR
    """
    device = get_device()
    
    # Get epochs to test
    epochs_param = request.form.get('epochs') or \
                  (request.json.get('epochs') if request.is_json else None)
    
    if epochs_param:
        if isinstance(epochs_param, str):
            epochs = [int(e.strip()) for e in epochs_param.split(',')]
        else:
            epochs = epochs_param
    else:
        # Get all available epochs
        checkpoints = list_checkpoints().json['checkpoints']
        epochs = [c['epoch'] for c in checkpoints if c['epoch'] is not None]
    
    # Get image
    if 'file' in request.files:
        file = request.files['file']
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    elif request.is_json and 'image' in request.json:
        original_img = base64_to_image(request.json['image'])
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    if original_img is None:
        return jsonify({'error': 'Could not decode image'}), 400
    
    original_size = original_img.shape[:2]
    
    try:
        segmenter = get_segmenter()
        
        # Original statistics
        original_masks = generate_mask(original_img, segmenter, device)
        original_ctr = calculate_ctr_from_masks(original_masks)
        
        # Heart dimensions
        heart_mask = original_masks['heart']
        heart_rows = np.where(heart_mask.sum(axis=1) > 0)[0]
        heart_cols = np.where(heart_mask.sum(axis=0) > 0)[0]
        heart_height = heart_rows[-1] - heart_rows[0] if len(heart_rows) > 0 else 0
        heart_width = heart_cols[-1] - heart_cols[0] if len(heart_cols) > 0 else 0
        heart_area = np.sum(heart_mask > 0)
        
        original_stats = {
            'ctr': float(round(original_ctr, 4)) if original_ctr else None,
            'is_cardiomegaly': bool(original_ctr >= 0.5) if original_ctr else None,
            'heart_width_px': int(heart_width),
            'heart_height_px': int(heart_height),
            'heart_area_px': int(heart_area),
            'image_size': list(original_size)
        }
        
        # Prepare for inpainting
        img_resized = cv2.resize(original_img, (512, 512))
        mask_resized = cv2.resize(heart_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        img_pil = Image.fromarray(cv2.cvtColor(
            np.stack([img_resized]*3, axis=-1),
            cv2.COLOR_BGR2RGB
        ))
        
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_resized, kernel, iterations=2)
        mask_pil = Image.fromarray(mask_dilated)
        
        # Test each epoch
        results = []
        prompt = "chest xray, healthy normal heart, clear lungs, medical imaging, high quality"
        negative_prompt = "enlarged heart, cardiomegaly, artifacts, blur, noise"
        
        for epoch in sorted(epochs):
            pipe = get_inpainting_pipe(epoch)
            if pipe is None:
                results.append({
                    'epoch': epoch,
                    'error': 'Checkpoint not found'
                })
                continue
            
            try:
                with torch.autocast(device):
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=img_pil,
                        mask_image=mask_pil,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        generator=torch.Generator(device).manual_seed(42)
                    ).images[0]
                
                result_gray = np.array(result.convert('L'))
                result_full = cv2.resize(result_gray, (original_size[1], original_size[0]))
                
                # Calculate result statistics
                result_masks = generate_mask(result_full, segmenter, device)
                result_ctr = calculate_ctr_from_masks(result_masks)
                
                result_heart = result_masks['heart']
                result_heart_area = np.sum(result_heart > 0)
                
                results.append({
                    'epoch': epoch,
                    'ctr': float(round(result_ctr, 4)) if result_ctr else None,
                    'is_healthy': bool(result_ctr < 0.5) if result_ctr else None,
                    'ctr_change': float(round(original_ctr - result_ctr, 4)) if original_ctr and result_ctr else None,
                    'heart_area_change_pct': float(round((result_heart_area - heart_area) / heart_area * 100, 2)) if heart_area > 0 else None,
                    'result_image': image_to_base64(result_full)
                })
                
            except Exception as e:
                results.append({
                    'epoch': epoch,
                    'error': str(e)
                })
        
        # Find best epoch
        valid_results = [r for r in results if r.get('ctr') is not None]
        best_result = min(valid_results, key=lambda x: x['ctr']) if valid_results else None
        
        return jsonify({
            'original': original_stats,
            'original_image': image_to_base64(original_img),
            'mask': image_to_base64(heart_mask),
            'results': results,
            'best_epoch': best_result['epoch'] if best_result else None,
            'best_ctr': best_result['ctr'] if best_result else None,
            'epochs_tested': len(epochs)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get overall project statistics."""
    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models" / "inpainting"
    
    stats = {
        'dataset': {
            'cardiomegaly_images': len(list((data_dir / "raw" / "cardiomegaly").glob("*.png"))) if (data_dir / "raw" / "cardiomegaly").exists() else 0,
            'healthy_images': len(list((data_dir / "raw" / "healthy").glob("*.png"))) if (data_dir / "raw" / "healthy").exists() else 0,
            'train_images': len(list((data_dir / "processed" / "train" / "images").glob("*.png"))) if (data_dir / "processed" / "train" / "images").exists() else 0,
            'val_images': len(list((data_dir / "processed" / "val" / "images").glob("*.png"))) if (data_dir / "processed" / "val" / "images").exists() else 0,
        },
        'model': {
            'device': get_device(),
            'base_model': 'runwayml/stable-diffusion-inpainting',
            'lora_rank': 16,
            'lora_alpha': 32,
        },
        'checkpoints': list_checkpoints().json
    }
    
    return jsonify(stats)


if __name__ == '__main__':
    print("=" * 60)
    print("Cardiac Inpainting API")
    print("=" * 60)
    print(f"Device: {get_device()}")
    print(f"Project root: {PROJECT_ROOT}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  GET  /health      - Health check")
    print("  GET  /checkpoints - List available checkpoints")
    print("  GET  /stats       - Project statistics")
    print("  POST /inpaint     - Inpaint cardiomegaly image")
    print("  POST /evaluate    - Evaluate with detailed stats")
    print("=" * 60)
    
    # Run without debug/reloader to avoid issues
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
