"""
Configuration file for CNN-based Phased Array Antenna Compensation
"""

import os

# =============================================================================
# Data Configuration
# =============================================================================
# Path to dataset directory containing CST output files
# Each file should be named with phase values: phase(0,0)=0.phase(0,1)=X... .txt
DATA_DIR = "./data"

# Output directory for results
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pattern dimensions (from CST simulation)
PATTERN_SIZE = 64  # 64x64 radiation pattern grid

# Number of elements in 4x4 array
ARRAY_SIZE = 4
NUM_ELEMENTS = ARRAY_SIZE * ARRAY_SIZE  # 16 total elements

# Element indices
REFERENCE_ELEMENT = (0, 0)  # Fixed reference element (phase = 0)
FAULTY_ELEMENT = (2, 1)     # Faulty element (phase = 0, amplitude = 0)

# Number of active elements (output dimension)
# 16 total - 1 reference - 1 faulty = 14 output neurons
NUM_ACTIVE_ELEMENTS = 14

# =============================================================================
# Data Preprocessing Configuration
# =============================================================================
# Gain values below this threshold are considered "noise floor"
NOISE_FLOOR_DBI = -50.0

# Normalization method: 'minmax' or 'standard'
NORMALIZATION_METHOD = 'minmax'

# Train/Validation/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# CNN Model Configuration
# =============================================================================
# Input shape: (64, 64, 1) - single channel grayscale radiation pattern
INPUT_SHAPE = (PATTERN_SIZE, PATTERN_SIZE, 1)

# Output shape: (14,) - phase values for 14 active elements
OUTPUT_SHAPE = (NUM_ACTIVE_ELEMENTS,)

# Phase range [0, 2*pi] for output normalization
PHASE_MIN = 0.0
PHASE_MAX = 2 * 3.14159265359  # 2*pi radians

# =============================================================================
# Training Configuration
# =============================================================================
# Batch size
BATCH_SIZE = 16

# Number of training epochs
EPOCHS = 150

# Learning rate
LEARNING_RATE = 0.001

# Optimizer: 'adam', 'sgd', 'rmsprop'
OPTIMIZER = 'adam'

# Loss function: 'mse', 'mae', 'huber'
LOSS_FUNCTION = 'mse'

# Metrics to track
METRICS = ['mae', 'mse']

# =============================================================================
# Callbacks Configuration
# =============================================================================
# Early stopping
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MODE = 'min'

# Learning rate reduction
LR_REDUCTION_PATIENCE = 15
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_MONITOR = 'val_loss'
LR_REDUCTION_MODE = 'min'
LR_REDUCTION_MIN_LR = 1e-7

# Model checkpoint
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_MODE = 'min'
CHECKPOINT_SAVE_BEST_ONLY = True

# =============================================================================
# Visualization Configuration
# =============================================================================
# Figure DPI for saved plots
FIGURE_DPI = 300

# Colormap for radiation patterns
PATTERN_CMAP = 'jet'

# Font size for plots
FONT_SIZE = 12

# =============================================================================
# Hardware Configuration
# =============================================================================
# Use GPU if available
USE_GPU = True

# Mixed precision training
MIXED_PRECISION = False
