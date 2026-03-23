"""
Data Loader and Preprocessing Module for CNN-based PAA Compensation

This module handles:
- Loading CST Microwave Studio far-field data files
- Parsing phase configurations from filenames
- Converting dB gain to linear scale
- Normalizing radiation patterns
- Creating train/validation/test splits
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import config


def parse_phase_from_filename(filename: str) -> Optional[np.ndarray]:
    """
    Parse phase values from CST output filename.
    
    Filename format: phase(0,0)=0.phase(0,1)=X.phase(0,2)=Y... .txt
    
    Args:
        filename: Name of the CST output file
        
    Returns:
        numpy array of 14 phase values (in radians) for active elements,
        or None if parsing fails
    """
    # Extract all phase values from filename
    pattern = r'phase\((\d),(\d)\)=(\d+(?:\.\d+)?)'
    matches = re.findall(pattern, filename)
    
    if not matches:
        return None
    
    # Create 4x4 phase matrix
    phase_matrix = np.zeros((4, 4))
    
    for row, col, phase_val in matches:
        phase_matrix[int(row), int(col)] = float(phase_val)
    
    # Extract active elements (excluding reference and faulty)
    # Reference: (0,0), Faulty: (2,1)
    active_phases = []
    
    for i in range(4):
        for j in range(4):
            if (i, j) != config.REFERENCE_ELEMENT and (i, j) != config.FAULTY_ELEMENT:
                # Convert degrees to radians
                active_phases.append(np.deg2rad(phase_matrix[i, j]))
    
    return np.array(active_phases)


def load_cst_pattern(filepath: str, target_size: int = 64) -> Optional[np.ndarray]:
    """
    Load and preprocess CST Microwave Studio far-field pattern.
    
    The CST file format:
    - Header line with column names
    - Separator line
    - Data rows: u, v, gain_dBi
    
    Args:
        filepath: Path to CST output file
        target_size: Target grid size for interpolation (default 64x64)
        
    Returns:
        2D numpy array of shape (target_size, target_size) containing
        normalized linear gain values, or None if loading fails
    """
    try:
        # Read the file, skipping header lines
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find where data starts (after separator line)
        data_start = 0
        for i, line in enumerate(lines):
            if '---' in line:
                data_start = i + 1
                break
        
        # Parse data lines
        u_vals = []
        v_vals = []
        gain_db = []
        
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                u_vals.append(float(parts[0]))
                v_vals.append(float(parts[1]))
                gain_db.append(float(parts[2]))
        
        u_vals = np.array(u_vals)
        v_vals = np.array(v_vals)
        gain_db = np.array(gain_db)
        
        # Replace noise floor values with minimum valid value
        gain_db = np.where(gain_db < config.NOISE_FLOOR_DBI, 
                          config.NOISE_FLOOR_DBI, gain_db)
        
        # Convert dB to linear scale
        gain_linear = 10 ** (gain_db / 10)
        
        # Create regular grid for interpolation
        u_grid = np.linspace(-1, 1, target_size)
        v_grid = np.linspace(-1, 1, target_size)
        U, V = np.meshgrid(u_grid, v_grid)
        
        # Interpolate onto regular grid using cubic interpolation
        points = np.column_stack((u_vals, v_vals))
        gain_grid = griddata(points, gain_linear, (U, V), method='cubic')
        
        # Handle any NaN values (outside convex hull) with nearest neighbor
        nan_mask = np.isnan(gain_grid)
        if np.any(nan_mask):
            gain_grid_nearest = griddata(points, gain_linear, (U, V), method='nearest')
            gain_grid[nan_mask] = gain_grid_nearest[nan_mask]
        
        return gain_grid
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def normalize_pattern(pattern: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize radiation pattern.
    
    Args:
        pattern: 2D array of gain values
        method: 'minmax' or 'standard'
        
    Returns:
        Normalized pattern
    """
    if method == 'minmax':
        # Min-max normalization to [0, 1]
        p_min = np.min(pattern)
        p_max = np.max(pattern)
        if p_max > p_min:
            return (pattern - p_min) / (p_max - p_min)
        else:
            return np.zeros_like(pattern)
    
    elif method == 'standard':
        # Z-score normalization
        p_mean = np.mean(pattern)
        p_std = np.std(pattern)
        if p_std > 0:
            return (pattern - p_mean) / p_std
        else:
            return np.zeros_like(pattern)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def load_dataset(data_dir: str, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load complete dataset from directory.
    
    Args:
        data_dir: Directory containing CST output files
        max_samples: Maximum number of samples to load (None = all)
        
    Returns:
        Tuple of (patterns, phases, filenames)
        - patterns: numpy array of shape (N, 64, 64, 1)
        - phases: numpy array of shape (N, 14)
        - filenames: list of source filenames
    """
    # Find all CST output files
    file_pattern = os.path.join(data_dir, "phase*.txt")
    filepaths = sorted(glob.glob(file_pattern))
    
    if max_samples is not None:
        filepaths = filepaths[:max_samples]
    
    print(f"Found {len(filepaths)} data files")
    
    patterns = []
    phases = []
    filenames = []
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        
        # Parse phase values from filename
        phase_vals = parse_phase_from_filename(filename)
        if phase_vals is None:
            print(f"Warning: Could not parse phases from {filename}")
            continue
        
        # Load radiation pattern
        pattern = load_cst_pattern(filepath, target_size=config.PATTERN_SIZE)
        if pattern is None:
            print(f"Warning: Could not load pattern from {filename}")
            continue
        
        # Normalize pattern
        pattern_norm = normalize_pattern(pattern, method=config.NORMALIZATION_METHOD)
        
        patterns.append(pattern_norm)
        phases.append(phase_vals)
        filenames.append(filename)
    
    # Convert to numpy arrays
    patterns = np.array(patterns)
    phases = np.array(phases)
    
    # Add channel dimension: (N, 64, 64) -> (N, 64, 64, 1)
    patterns = np.expand_dims(patterns, axis=-1)
    
    print(f"Successfully loaded {len(patterns)} samples")
    print(f"Pattern shape: {patterns.shape}")
    print(f"Phase shape: {phases.shape}")
    
    return patterns, phases, filenames


def split_dataset(patterns: np.ndarray, phases: np.ndarray, 
                  train_ratio: float = 0.7, val_ratio: float = 0.15,
                  random_seed: int = 42) -> Tuple:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        patterns: Input patterns array
        phases: Output phases array
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        patterns, phases, 
        test_size=(1 - train_ratio - val_ratio),
        random_state=random_seed
    )
    
    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_seed
    )
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(X_train)} samples ({len(X_train)/len(patterns)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(patterns)*100:.1f}%)")
    print(f"  Test:       {len(X_test)} samples ({len(X_test)/len(patterns)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 16, shuffle: bool = True):
    """
    Create a data generator for training.
    
    Args:
        X: Input patterns
        y: Output phases
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of (X_batch, y_batch)
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]


def save_dataset_info(output_path: str, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save dataset information to file.
    
    Args:
        output_path: Path to save info file
        X_train, X_val, X_test: Input arrays
        y_train, y_val, y_test: Output arrays
    """
    with open(output_path, 'w') as f:
        f.write("Dataset Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training samples:   {len(X_train)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples:       {len(X_test)}\n")
        f.write(f"Total samples:      {len(X_train) + len(X_val) + len(X_test)}\n\n")
        f.write(f"Input shape:  {X_train.shape[1:]}\n")
        f.write(f"Output shape: {y_train.shape[1:]}\n\n")
        f.write("Phase statistics (training set, radians):\n")
        f.write(f"  Min:  {np.min(y_train):.4f}\n")
        f.write(f"  Max:  {np.max(y_train):.4f}\n")
        f.write(f"  Mean: {np.mean(y_train):.4f}\n")
        f.write(f"  Std:  {np.std(y_train):.4f}\n")


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    patterns, phases, filenames = load_dataset(config.DATA_DIR)
    
    if len(patterns) > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            patterns, phases, 
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            random_seed=config.RANDOM_SEED
        )
        
        # Save dataset info
        save_dataset_info(
            os.path.join(config.OUTPUT_DIR, "dataset_info.txt"),
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        print("\nData loading test completed successfully!")
