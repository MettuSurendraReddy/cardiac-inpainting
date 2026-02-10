# Cardiac Inpainting API

Flask API for the Cardiac Inpainting pipeline.

## Setup

```bash
pip install flask flask-cors
```

## Run the API

```bash
cd api
python app.py
```

Or with production server:
```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

## Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "timestamp": "2026-02-04T10:00:00"
}
```

### GET /checkpoints
List all available model checkpoints.

**Response:**
```json
{
  "checkpoints": [
    {"name": "checkpoint_epoch_40", "epoch": 40, "path": "..."},
    {"name": "checkpoint_epoch_50", "epoch": 50, "path": "..."},
    {"name": "best_model", "epoch": null, "path": "..."}
  ],
  "total": 3
}
```

### GET /stats
Get project statistics.

**Response:**
```json
{
  "dataset": {
    "cardiomegaly_images": 626,
    "healthy_images": 626,
    "train_images": 500,
    "val_images": 126
  },
  "model": {
    "device": "cuda",
    "base_model": "runwayml/stable-diffusion-inpainting",
    "lora_rank": 16
  },
  "checkpoints": {...}
}
```

### POST /inpaint
Inpaint a cardiomegaly image to generate a healthy version.

**Request (multipart form):**
```
file: <image file>
epoch: 50  (optional, default: best_model)
num_candidates: 3  (optional, default: 1)
return_format: base64  (optional: 'base64' or 'file')
```

**Request (JSON):**
```json
{
  "image": "<base64 encoded image>",
  "epoch": 50,
  "num_candidates": 3,
  "return_format": "base64"
}
```

**Response:**
```json
{
  "success": true,
  "original_ctr": 0.5432,
  "result_ctr": 0.4521,
  "ctr_reduction": 0.0911,
  "epoch_used": 50,
  "num_candidates": 3,
  "result_image": "<base64 encoded image>",
  "mask": "<base64 encoded mask>",
  "mask_dilated": "<base64 encoded dilated mask>"
}
```

**Example with curl:**
```bash
# Using file upload
curl -X POST -F "file=@cardiomegaly.png" -F "epoch=50" http://localhost:5000/inpaint

# Using base64
curl -X POST -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "epoch": 50}' \
  http://localhost:5000/inpaint
```

### POST /evaluate
Evaluate model across multiple epochs with detailed statistics.

**Request (multipart form):**
```
file: <image file>
epochs: 40,50,60,70,80  (optional, comma-separated)
```

**Request (JSON):**
```json
{
  "image": "<base64 encoded image>",
  "epochs": [40, 50, 60, 70, 80]
}
```

**Response:**
```json
{
  "original": {
    "ctr": 0.5432,
    "is_cardiomegaly": true,
    "heart_width_px": 312,
    "heart_height_px": 285,
    "heart_area_px": 70521,
    "image_size": [1024, 1024]
  },
  "original_image": "<base64>",
  "mask": "<base64>",
  "results": [
    {
      "epoch": 40,
      "ctr": 0.5102,
      "is_healthy": false,
      "ctr_change": 0.0330,
      "heart_area_change_pct": -8.5,
      "result_image": "<base64>"
    },
    {
      "epoch": 50,
      "ctr": 0.4521,
      "is_healthy": true,
      "ctr_change": 0.0911,
      "heart_area_change_pct": -15.2,
      "result_image": "<base64>"
    }
  ],
  "best_epoch": 50,
  "best_ctr": 0.4521,
  "epochs_tested": 2
}
```

**Example with curl:**
```bash
curl -X POST -F "file=@cardiomegaly.png" -F "epochs=40,50,60" \
  http://localhost:5000/evaluate
```

## Python Client Example

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with open('cardiomegaly.png', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Inpaint
response = requests.post(
    'http://localhost:5000/inpaint',
    json={
        'image': image_b64,
        'epoch': 50,
        'num_candidates': 3
    }
)

result = response.json()
print(f"Original CTR: {result['original_ctr']}")
print(f"Result CTR: {result['result_ctr']}")
print(f"Success: {result['success']}")

# Decode result image
result_bytes = base64.b64decode(result['result_image'])
result_image = Image.open(io.BytesIO(result_bytes))
result_image.save('healthy_result.png')
```

## Notes

- First request will be slower as models are loaded into memory
- Subsequent requests use cached models
- Each epoch checkpoint is cached separately
- For production, use a single worker (`-w 1`) to share model cache
