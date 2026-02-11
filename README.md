# Cardiac Inpainting: Cardiomegaly to Healthy Heart Transformation

Transform chest X-rays with **cardiomegaly** (enlarged heart) into realistic X-rays showing a **healthy heart** using Stable Diffusion inpainting with LoRA fine-tuning.

## Overview

This project implements a **counterfactual medical image generation** system that transforms chest X-ray images with cardiomegaly into anatomically plausible healthy versions. The transformation is minimal: only the heart region changes while preserving all other anatomical structures.

**Important**: This is a research project and should not be used for clinical diagnosis or treatment decisions.

## Pipeline Components

1. **Segmenter**: CheXMask HybridGNet for cardiac and lung segmentation
2. **Inpainter**: Stable Diffusion Inpainting model fine-tuned with LoRA on healthy chest X-rays
3. **Classifier**: Binary classifier (healthy vs cardiomegaly) - supports multiple architectures (ResNet, DenseNet, EfficientNet)
4. **Validator**: Anatomical validation using Cardiothoracic Ratio (CTR) measurements

```
Input X-ray (Cardiomegaly)
         │
         ▼
┌─────────────────┐
│   Segmenter     │  CheXMask HybridGNet
│  (Heart Mask)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Inpainter     │  Stable Diffusion + LoRA
│  (Generate)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │  CTR Check + Classifier
│                 │
└────────┬────────┘
         │
         ▼
Output X-ray (Healthy)
```

## Key Concept: Cardiothoracic Ratio (CTR)

The **Cardiothoracic Ratio (CTR)** is a clinical measurement used to assess heart size:

```
CTR = Heart Width / Chest Width
```

- **CTR < 0.5**: Healthy heart (normal size)
- **CTR >= 0.5**: Cardiomegaly (enlarged heart)

The system transforms images with CTR >= 0.5 to images with CTR < 0.5, validated through automatic CTR calculation using segmented heart and lung regions.

## Training Strategy

The inpainting model is trained **exclusively on healthy chest X-rays** with dilated masks:

1. Take a healthy chest X-ray
2. Generate the heart mask using CheXMask segmentation
3. **Dilate the mask by 30-80%** to simulate a larger (cardiomegaly-sized) region
4. Train the model to reconstruct the original healthy heart within this enlarged mask

This approach teaches the model to fill oversized masks with correctly-sized healthy hearts, without requiring paired cardiomegaly-to-healthy data.

## Dataset

This project uses the [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC).

### Dataset Filtering

From the full NIH dataset, we filter:

- **View Position**: Only PA (Posteroanterior) X-rays
- **Labels**: Only "No Finding" (healthy) and "Cardiomegaly"
- **One image per patient** to reduce data leakage

### Data Split

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 500 (healthy only) | Train inpainting model |
| Validation | 126 (healthy only) | Monitor training |
| Test | Cardiomegaly images | Evaluate transformation |

## Model Configuration

### LoRA Settings

```yaml
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules:
    - "to_q"
    - "to_v"
    - "to_k"
    - "to_out.0"
```

### Training Configuration

```yaml
training:
  batch_size: 1          # Default for 8GB GPUs
  learning_rate: 0.0001
  num_epochs: 100
  gradient_accumulation_steps: 4
  mixed_precision: true
  warmup_steps: 500
```

### Validation Thresholds

```yaml
validation:
  min_ctr: 0.35          # Below = unrealistically small heart
  max_ctr: 0.50          # Above = still cardiomegaly
  min_healthy_confidence: 0.80
```

### Inference Settings

```yaml
inference:
  num_inference_steps: 50
  guidance_scale: 7.5
  num_candidates: 5      # Generate multiple, pick best
  max_attempts: 10
```

## Project Structure

```
cardiac-inpainting/
├── api/                        # Flask API
│   ├── app.py
│   └── README.md
├── configs/
│   ├── default.yaml
│   ├── training.yaml
│   └── inference.yaml
├── data/
│   ├── raw/
│   │   ├── cardiomegaly/
│   │   └── healthy/
│   ├── masks/
│   └── processed/
├── models/
│   ├── CheXmask-Database/      # Segmentation model
│   ├── classifier/
│   └── inpainting/             # LoRA checkpoints
├── outputs/
├── scripts/
│   ├── calibrate_classifier_temperature.py
│   ├── compare_checkpoints.py
│   ├── evaluate.py
│   ├── export_nih_dataset_a_to_data_raw.py
│   ├── generate_masks.py
│   ├── inference.py
│   ├── make_evaluation_pictures.py
│   ├── prepare_data.py
│   ├── test_inference.py
│   ├── train.py
│   └── train_dataset_a_classifier.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── augmentation.py
│   │   ├── dataset.py
│   │   └── preparation.py
│   ├── inference/
│   │   ├── batch_processor.py
│   │   └── pipeline.py
│   ├── models/
│   │   ├── classifier.py
│   │   ├── inpainter.py
│   │   └── segmenter.py
│   ├── training/
│   │   ├── losses.py
│   │   └── trainer.py
│   └── validation/
│       ├── anatomical.py
│       └── metrics.py
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with 8GB+ VRAM
- ~50GB storage for dataset and models

### Installation

```bash
git clone https://github.com/your-repo/cardiac-inpainting.git
cd cardiac-inpainting

python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

pip install -r requirements.txt
pip install -e .
```

### Data Preparation

1. **Download NIH ChestX-ray14 Dataset** from [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. **Export Dataset A** (PA-only, balanced classes)
   ```bash
   python scripts/export_nih_dataset_a_to_data_raw.py \
       --data-entry-csv content/Data_Entry_2017.csv \
       --images-root content \
       --out-root data
   ```

3. **Generate heart masks**
   ```bash
   python scripts/generate_masks.py \
       --input-dir data/raw \
       --output-dir data/masks \
       --category healthy
   ```

4. **Prepare training data**
   ```bash
   python scripts/prepare_data.py \
       --images-dir data/raw/healthy \
       --masks-dir data/masks/healthy \
       --output-dir data/processed
   ```

### Model Setup

1. **CheXMask**: Download from [PhysioNet](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/) and place in `models/CheXmask-Database/`

2. **Train classifier** (optional, for validation)
   ```bash
   python scripts/train_dataset_a_classifier.py \
       --device cuda \
       --epochs 30 \
       --model resnet50
   ```

3. **Stable Diffusion**: Downloaded automatically from HuggingFace on first run

### Training

```bash
python scripts/train.py \
    --config configs/training.yaml \
    --data-dir data/processed \
    --output-dir models/inpainting \
    --epochs 100
```

Training automatically resumes from the latest checkpoint if interrupted.

### Inference

**Single image:**
```bash
python scripts/inference.py \
    --input /path/to/cardiomegaly_xray.png \
    --output /path/to/output.png \
    --checkpoint models/inpainting/final_model
```

**Batch processing:**
```bash
python scripts/inference.py \
    --input data/raw/cardiomegaly/ \
    --output outputs/generated/ \
    --checkpoint models/inpainting/final_model \
    --save-comparison
```

**Compare checkpoints across epochs:**
```bash
python scripts/compare_checkpoints.py \
    --image data/raw/cardiomegaly/example.png \
    --epochs 40,50,60,70,80
```

### Classifier Calibration (Recommended)

Neural network softmax scores are often overconfident. Use temperature scaling:

```bash
python scripts/calibrate_classifier_temperature.py --device cuda
```

The calibration is saved to `outputs/classifier/dataset_a_calibration.json` and loaded automatically during inference.

### Generate Evaluation Pictures

```bash
python scripts/make_evaluation_pictures.py \
    --input-dir data/raw/cardiomegaly \
    --checkpoint models/inpainting/final_model \
    --best-num 4 --worst-num 4 --random-num 4 \
    --output-dir Evaluation_pictures
```

### API Server

```bash
cd api
python app.py
```

Endpoints:
- `GET /health` - Health check
- `GET /checkpoints` - List available checkpoints
- `POST /inpaint` - Inpaint a single image
- `POST /evaluate` - Evaluate across multiple epochs

## Results

Evaluation on cardiomegaly test images:

| Metric | Value |
|--------|-------|
| Success Rate | 75.1% (187/249) |
| Mean CTR (input) | 0.54 |
| Mean CTR (output) | 0.45 |
| Mean SSIM (outside mask) | 0.92 |
| Mean Healthy Confidence | 95% |

## Limitations

- This is a research project, not a clinical tool
- Success rate drops for severe cardiomegaly (CTR > 0.60)
- Models trained for too many epochs may produce artifacts (dark lines across heart region)
- Best results typically from checkpoints around epoch 40-50, not later epochs
- ~25% of images fail validation and cannot be transformed

## License

MIT License

## References

- Wang et al. (2017). ChestX-ray8: Hospital-scale Chest X-ray Database. CVPR.
- Gaggion et al. (2023). CheXMask Database. PhysioNet.
- Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv.
