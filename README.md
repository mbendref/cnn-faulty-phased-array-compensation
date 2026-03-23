# CNN-Based Compensation of Faulty Planar Phased-Array Radiation Patterns

This repository contains the complete implementation of the CNN-based approach for rapid compensation of faulty planar phased-array antennas, as described in the paper "CNN-Based Compensation of Faulty Planar Phased-Array Radiation Patterns" submitted to Scientific Reports.

## Overview

Traditional compensation methods (Genetic Algorithms, Particle Swarm Optimization) require 120-300 seconds per failure event. This CNN-based approach achieves **sub-degree accuracy** with **~200 ms inference time** on a standard GPU - a **600-1500x speedup**.

### Key Features

- **Input**: 64x64 radiation pattern image from faulty 4x4 array
- **Output**: 14 phase values for active elements (excluding reference and faulty)
- **Accuracy**: MAE = 6.1×10⁻³ rad (0.35°), R² = 0.98
- **Speed**: ~200 ms inference time on NVIDIA RTX 3060
- **Architecture**: 4 convolutional blocks + 3 dense layers (~2.8M parameters)

## Repository Structure

```
cnn_paa_code/
├── config.py           # Configuration parameters
├── data_loader.py      # Data preprocessing and loading
├── model.py            # CNN architecture definition
├── train.py            # Training script
├── evaluate.py         # Evaluation and visualization
├── inference.py        # Inference and compensation
├── utils.py            # Utility functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (for GPU support)
- cuDNN 8.6+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd cnn_paa_code
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Support

For NVIDIA GPU acceleration:
```bash
pip install tensorflow[and-cuda]
```

Verify GPU availability:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Data Format

### Input Data

The training data consists of CST Microwave Studio far-field simulation outputs. Each file should be named with phase values:

```
phase(0,0)=0.phase(0,1)=X.phase(0,2)=Y... .txt
```

Where:
- `phase(i,j)` is the phase of element at row i, column j (in degrees)
- Element (0,0) is the fixed reference (phase = 0)
- Element (2,1) is the faulty element (phase = 0, amplitude = 0)

### File Format

Each CST output file contains:
```
u             v             Abs(Grlz)[dBi   ]  normalized
----------------------------------------------------------------
 -1.0000   -1.0000  -259.230
 -0.9830   -1.0000  -259.230
 ...
```

Columns:
- `u`: Direction cosine u = sin(θ)cos(φ)
- `v`: Direction cosine v = sin(θ)sin(φ)
- `Abs(Grlz)`: Gain in dBi

## Usage

### 1. Data Preparation

Place your CST output files in the `data/` directory:

```
data/
├── phase(0,0)=0.phase(0,1)=45.phase(0,2)=90... .txt
├── phase(0,0)=0.phase(0,1)=30.phase(0,2)=60... .txt
└── ...
```

Update `config.py` if your data is in a different location:
```python
DATA_DIR = "./data"  # Your data directory
```

### 2. Training

Run the training script:

```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Split into train/validation/test sets (70%/15%/15%)
- Train the CNN model
- Save the best model to `output/best_model.h5`
- Generate training plots and history

Training configuration can be modified in `config.py`:
```python
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001
```

### 3. Evaluation

Evaluate the trained model:

```bash
python evaluate.py
```

This generates:
- Test set metrics (MAE, RMSE, R²)
- Per-element accuracy plots (Figure 6)
- Training curves (Figure 4)
- Overfitting analysis (Figure 5)

### 4. Inference

Run inference on new patterns:

```bash
python inference.py
```

Example usage in Python:
```python
from inference import load_model, compensate_pattern
import numpy as np

# Load model
model = load_model("output/best_model.h5")

# Load your pattern (64x64 array)
pattern = np.load("your_pattern.npy")

# Compensate
phases, inference_time = compensate_pattern(model, pattern)

print(f"Predicted phases: {phases}")
print(f"Inference time: {inference_time*1000:.2f} ms")
```

## Model Architecture

```
Input (64x64x1)
    |
    v
[Conv Block 1]  Conv2D(64, 7x7) -> BN -> ReLU -> Conv2D(64, 3x3) -> BN -> ReLU
    |           MaxPool(2x2) -> Dropout(0.10)
    v
[Conv Block 2]  Conv2D(128, 5x5) -> BN -> ReLU -> Conv2D(128, 3x3) -> BN -> ReLU
    |           MaxPool(2x2) -> Dropout(0.15)
    v
[Conv Block 3]  Conv2D(256, 3x3) -> BN -> ReLU -> Conv2D(256, 3x3) -> BN -> ReLU
    |           MaxPool(2x2) -> Dropout(0.20)
    v
[Conv Block 4]  Conv2D(512, 3x3) -> BN -> ReLU -> Conv2D(512, 3x3) -> BN -> ReLU
    |           GlobalAveragePooling2D -> Dropout(0.30)
    v
[Dense Layers]  Dense(1024) -> BN -> ReLU -> Dropout(0.40)
    |           Dense(256) -> ReLU -> Dropout(0.20)
    v
Output (14)     Linear activation
```

Total parameters: ~2.8 million

## Performance Metrics

### Phase Prediction Accuracy

| Metric | Value |
|--------|-------|
| MAE | 6.1×10⁻³ rad (0.35°) |
| RMSE | 6.6×10⁻³ rad (0.38°) |
| R² | 0.98 |

### Computational Efficiency

| Method | Time | Speedup |
|--------|------|---------|
| Genetic Algorithm | 180-300 s | 1x |
| Particle Swarm Optimization | 120-240 s | 1.5x |
| **CNN Inference (Proposed)** | **~200 ms** | **600-1500x** |

### Pattern Compensation

- Average RMSE improvement: 32.1% (±3.8%)
- Range: 28.28% to 35.91% improvement

## Output Files

After training, the following files are generated in `output/`:

```
output/
├── best_model.h5              # Best model checkpoint
├── final_model.h5             # Final model after training
├── training_history.json      # Training metrics
├── training_history.png       # Training curves (Figure 4)
├── learning_rate.png          # LR schedule
├── training_info.txt          # Training summary
├── dataset_info.txt           # Dataset statistics
├── test_results.txt           # Test set evaluation
└── evaluation/
    ├── test_metrics.txt       # Detailed test metrics
    ├── per_element_accuracy.png  # Figure 6
    ├── training_curves.png    # Figure 4
    └── overfitting_analysis.png  # Figure 5
```

## Configuration

All parameters can be modified in `config.py`:

### Data Parameters
```python
PATTERN_SIZE = 64           # Input pattern dimensions
NUM_ACTIVE_ELEMENTS = 14    # Output dimension
TRAIN_RATIO = 0.70          # Training set ratio
VAL_RATIO = 0.15            # Validation set ratio
```

### Model Parameters
```python
INPUT_SHAPE = (64, 64, 1)   # Input shape
OUTPUT_SHAPE = (14,)        # Output shape
```

### Training Parameters
```python
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mse'
```

### Callback Parameters
```python
EARLY_STOPPING_PATIENCE = 25
LR_REDUCTION_PATIENCE = 15
LR_REDUCTION_FACTOR = 0.5
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{djassem2026cnn,
  title={CNN-Based Compensation of Faulty Planar Phased-Array Radiation Patterns},
  author={Djassem, Bendref Mansour and Challal, Mouloud and Staraj, Robert and Abdolmohammadi, Hamid R. and Khosravi, Nima and Oubelaid, Adel},
  journal={Scientific Reports},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## License

This code is provided for academic research purposes. No restrictions apply for academic use.

## Contact

For questions or issues, please contact:
- Bendref Mansour Djassem ( author)
- Signals and System Laboratory, University M'Hamed BOUGARA- Boumerdes, Algeria

## Acknowledgments

- CST Microwave Studio for electromagnetic simulations
- TensorFlow/Keras team for the deep learning framework
