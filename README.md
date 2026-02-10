# Cardiac Inpainting: Cardiomegaly to Healthy Heart Transformation

Transform chest X-rays with **cardiomegaly** (enlarged heart) into realistic X-rays showing a **healthy heart** using Stable Diffusion inpainting with LoRA fine-tuning.

## Table of Contents

- [Overview](#overview)
- [Key Concept: Cardiothoracic Ratio (CTR)](#key-concept-cardiothoracic-ratio-ctr)
- [Methods & Architecture](#methods--architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Model Setup](#model-setup)
  - [Training](#training)
  - [Inference](#inference)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

## Overview

This project implements an AI-powered **counterfactual medical image generation** system that transforms chest X-ray images with cardiomegaly into anatomically plausible healthy versions. The transformation is minimal—only the heart region changes while preserving all other anatomical structures.

### Pipeline Components

1. **Classifier**: Pre-trained ResNet18 binary classifier (Cardiomegaly vs Healthy)
2. **Segmenter**: CheXMask HybridGNet for cardiac and lung segmentation
3. **Inpainter**: Stable Diffusion Inpainting model fine-tuned with LoRA on healthy chest X-rays
4. **Validator**: Anatomical validation using Cardiothoracic Ratio (CTR) measurements

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Input X-ray   │────▶│   Classifier    │────▶│  Cardiomegaly?  │
│  (Cardiomegaly) │     │   (ResNet18)    │     │   Yes / No      │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │ Yes
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Output X-ray   │◀────│   Inpainter     │◀────│   Segmenter     │
│    (Healthy)    │     │ (SD + LoRA)     │     │  (CheXMask)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        ▼                       │
┌─────────────────┐             │
│   Validator     │◀────────────┘
│  (CTR Check)    │
└─────────────────┘
```

## Key Concept: Cardiothoracic Ratio (CTR)

The **Cardiothoracic Ratio (CTR)** is a clinical measurement used to determine heart health:

```
CTR = Heart Width / Chest Width
```

- **CTR < 0.5**: Healthy heart (normal size)
- **CTR ≥ 0.5**: Cardiomegaly (enlarged heart)

The system transforms images with CTR ≥ 0.5 to images with CTR < 0.5, validated through automatic CTR calculation using segmented heart and lung regions.

## Methods & Architecture

### Training Strategy

The inpainting model is trained **exclusively on healthy chest X-rays**:

1. Take a healthy chest X-ray
2. Mask out the heart region using CheXMask segmentation
3. Train the model to reconstruct the original healthy heart
4. The model learns the distribution of "healthy hearts in anatomical context"

This approach ensures the model generates anatomically correct healthy hearts when applied to cardiomegaly images.

### Inpainting Model

- **Base Model**: [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) (`runwayml/stable-diffusion-inpainting`)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
- **Input**: Original image + binary heart mask
- **Output**: Image with inpainted healthy heart

### LoRA Configuration

```yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32           # Scaling factor
  lora_dropout: 0.05       # Regularization
  target_modules:          # Adapted layers
    - "to_q"
    - "to_v"
    - "to_k"
    - "to_out.0"
```

### Segmentation (CheXMask)

We use [CheXMask HybridGNet](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/) for accurate anatomical segmentation:

- Heart segmentation for inpainting mask generation
- Lung segmentation for CTR calculation

### Classification

A ResNet18 classifier trained on the NIH Chest X-ray dataset distinguishes cardiomegaly from healthy images, used for:
- Input verification (confirming cardiomegaly before processing)
- Output validation (ensuring generated image is classified as healthy)

## Dataset

This project uses the [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) from Kaggle.

### Dataset Filtering

From the full NIH dataset, we filtered:

1. **Conditions**: Only two classes:
   - `No Finding` → Healthy
   - `Cardiomegaly` → Cardiomegaly

2. **View Position**: Only **PA (Posteroanterior)** X-rays
   - PA views provide standardized frontal chest images
   - Lateral and AP views are excluded for consistency

### Data Split

- **Training**: Healthy images only (the model learns to generate healthy hearts)
- **Validation**: Both healthy and cardiomegaly images (for testing transformations)

## Project Structure

```
model-v4/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default settings
│   ├── training.yaml          # Training hyperparameters
│   └── inference.yaml         # Inference settings
│
├── data/                       # Data directory (not in repo)
│   ├── raw/                   # Original filtered X-rays
│   │   ├── cardiomegaly/     # Cardiomegaly images
│   │   └── healthy/          # Healthy images
│   ├── masks/                 # Generated heart masks
│   │   ├── cardiomegaly/
│   │   └── healthy/
│   └── processed/             # Processed training data
│       ├── train/
│       └── val/
│
├── models/                     # Model weights (not in repo)
│   ├── CheXmask-Database/     # CheXMask segmentation
│   ├── classifier/            # Cardiomegaly classifier
│   └── inpainting/            # Trained LoRA weights
│
│
├── outputs/                    # Generated outputs
│   ├── generated/             # Generated healthy images
│   ├── comparisons/           # Side-by-side comparisons
│   └── logs/                  # Training logs
│
├── scripts/                    # Executable scripts
│   ├── prepare_data.py        # Data preparation
│   ├── generate_masks.py      # Mask generation
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── inference.py           # Inference script
│
├── src/                        # Source code
│   ├── data/                  # Data loading & augmentation
│   ├── models/                # Model wrappers
│   ├── training/              # Training loop & losses
│   ├── inference/             # Inference pipeline
│   └── validation/            # Anatomical validation
│
├── tests/                      # Unit tests
├── requirements.txt
├── setup.py
└── README.md
```

## Getting Started

### Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (recommended)
- **Storage**: ~50GB for dataset and models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/cardiac-inpainting.git
   cd cardiac-inpainting
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package in development mode**
   ```bash
   pip install -e .
   ```

### Data Preparation

Since the dataset is not included in the repository, you need to prepare it manually:

1. **Download the NIH Chest X-ray Dataset**
   
   Download from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) (requires Kaggle account)
   
   ```bash
   # Using Kaggle CLI
   kaggle datasets download -d nih-chest-xrays/data
   ```

2. **Filter the dataset**
   
   Extract only PA X-rays with "No Finding" or "Cardiomegaly" labels:
   
   ```bash
   python scripts/prepare_data.py \
       --source-dir /path/to/nih-chest-xrays \
       --output-dir data/raw \
       --view-position PA \
       --conditions "No Finding" "Cardiomegaly"
   ```

3. **Generate heart masks**
   
   ```bash
   python scripts/generate_masks.py \
       --input-dir data/raw \
       --output-dir data/masks
   ```

4. **Prepare training data**
   
   ```bash
   python scripts/prepare_data.py \
       --images-dir data/raw/healthy \
       --masks-dir data/masks/healthy \
       --output-dir data/processed \
       --train-split 0.8
   ```

### Model Setup

#### 1. CheXMask Segmentation Model

Download the CheXMask HybridGNet weights:

1. Visit [PhysioNet CheXMask](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/)
2. Download the model weights
3. Place in `models/CheXmask-Database/Weights/`

#### 2. Cardiomegaly Classifier

Train or download a pre-trained ResNet18 classifier:

```bash
# If training your own classifier
python scripts/train_classifier.py \
    --data-dir data/raw \
   --output models/classifier/dataset_a_classifier.pt
```

Or place pre-trained weights in `models/classifier/dataset_a_classifier.pt`

#### 3. Stable Diffusion Inpainting

The base Stable Diffusion model is downloaded automatically from HuggingFace on first run.

### Training

Train the inpainting model with LoRA fine-tuning:

```bash
python scripts/train.py \
    --config configs/training.yaml \
    --data-dir data/processed \
    --output-dir models/inpainting \
    --epochs 100 \
    --batch-size 4
```

**Key training arguments:**
- `--resume`: Resume from a checkpoint
- `--wandb`: Enable Weights & Biases logging
- `--dry-run`: Test with one batch

Training will save checkpoints every 10 epochs and the best model based on validation loss.

### Inference

**Single image:**
```bash
python scripts/inference.py \
    --input /path/to/cardiomegaly_xray.png \
    --output /path/to/output.png \
    --checkpoint models/inpainting/best_model
```

**Batch processing:**
```bash
python scripts/inference.py \
    --input data/raw/cardiomegaly/ \
    --output outputs/generated/ \
    --checkpoint models/inpainting/best_model \
    --save-comparison
```

**Key inference arguments:**
- `--num-candidates`: Number of generation candidates (default: 5)
- `--no-classifier`: Skip classifier validation
- `--no-anatomical`: Skip CTR validation
- `--save-comparison`: Save before/after comparison images

#### Classifier confidence calibration (recommended)

The classifier's softmax confidence can be **overconfident**. This repo supports **temperature scaling** calibration.

1) Calibrate (writes a JSON report):

```bash
python scripts/calibrate_classifier_temperature.py --device cuda
```

2) Inference/evaluation/presentation scripts will automatically load the temperature from:
`outputs/classifier/dataset_a_calibration.json` (if present).

Optional overrides:
- `--classifier-temperature <T>`: force a specific temperature
- `--classifier-calibration <path>`: use a different calibration JSON
- `--disable-classifier-calibration`: disable auto-loading

## Configuration

Configuration files are in `configs/`:

- `default.yaml`: Base configuration
- `training.yaml`: Training-specific settings
- `inference.yaml`: Inference-specific settings

### Key Configuration Options

```yaml
# Training
training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
  mixed_precision: true

# Validation thresholds
validation:
  min_ctr: 0.35          # Minimum healthy CTR
  max_ctr: 0.50          # Maximum (below cardiomegaly)
  min_healthy_confidence: 0.80

# Inference
inference:
  num_inference_steps: 50
  guidance_scale: 7.5
  num_candidates: 5
```

## Results

The trained model successfully transforms cardiomegaly X-rays while:
- Maintaining anatomical plausibility (validated CTR)
- Preserving surrounding structures (lungs, ribs, etc.)
- Producing images classified as healthy by the classifier

Example results can be found in `outputs/comparisons/` after running inference.

## Technologies Used

- **PyTorch**: Deep learning framework
- **Diffusers**: Stable Diffusion implementation
- **PEFT**: LoRA fine-tuning
- **CheXMask HybridGNet**: Anatomical segmentation
- **torchvision**: Image processing and pre-trained models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- [CheXMask Database](https://physionet.org/content/chexmask-cxr-segmentation-data/0.4/)
- [Stable Diffusion](https://stability.ai/stable-diffusion)
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
