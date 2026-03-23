"""
Inference and Pattern Compensation Module

This module provides:
- Fast inference on new radiation patterns
- Pattern compensation workflow
- Comparison with original and faulty patterns
- Performance metrics for compensation
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from typing import Tuple, Optional

import config
from data_loader import load_cst_pattern, normalize_pattern
from evaluate import plot_cdf_comparison, plot_radiation_pattern


def load_model(model_path: str) -> keras.Model:
    """
    Load trained model for inference.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras Model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def predict_phases(model: keras.Model, pattern: np.ndarray) -> np.ndarray:
    """
    Predict phases from a radiation pattern.
    
    Args:
        model: Trained CNN model
        pattern: Radiation pattern (64, 64) or (64, 64, 1)
        
    Returns:
        Predicted phases (14,) in radians
    """
    # Ensure correct shape
    if len(pattern.shape) == 2:
        pattern = np.expand_dims(pattern, axis=-1)
    if len(pattern.shape) == 3:
        pattern = np.expand_dims(pattern, axis=0)
    
    # Predict
    phases = model.predict(pattern, verbose=0)
    
    return phases[0]


def compensate_pattern(model: keras.Model, 
                       target_pattern: np.ndarray,
                       measure_time: bool = True) -> Tuple[np.ndarray, Optional[float]]:
    """
    Compensate a faulty array pattern to match target pattern.
    
    This is the inference/compensation workflow described in the paper:
    1. Input: Target pattern (original intact array pattern)
    2. CNN predicts phases for 14 active elements
    3. Apply phases to faulty array to recover pattern
    
    Args:
        model: Trained CNN model
        target_pattern: Target radiation pattern (64, 64) - the desired pattern
        measure_time: Whether to measure inference time
        
    Returns:
        Tuple of (predicted_phases, inference_time)
    """
    # Normalize pattern
    pattern_norm = normalize_pattern(target_pattern, method=config.NORMALIZATION_METHOD)
    
    # Measure inference time
    if measure_time:
        # Warm-up
        _ = predict_phases(model, pattern_norm)
        
        # Timed run
        start_time = time.time()
        phases = predict_phases(model, pattern_norm)
        inference_time = time.time() - start_time
    else:
        phases = predict_phases(model, pattern_norm)
        inference_time = None
    
    return phases, inference_time


def reconstruct_full_phases(active_phases: np.ndarray) -> np.ndarray:
    """
    Reconstruct full 4x4 phase matrix from 14 active phases.
    
    Args:
        active_phases: 14 phase values for active elements
        
    Returns:
        4x4 phase matrix with reference and faulty elements set to 0
    """
    full_phases = np.zeros((4, 4))
    
    idx = 0
    for i in range(4):
        for j in range(4):
            if (i, j) != config.REFERENCE_ELEMENT and (i, j) != config.FAULTY_ELEMENT:
                full_phases[i, j] = active_phases[idx]
                idx += 1
    
    return full_phases


def batch_compensate(model: keras.Model, 
                     target_patterns: np.ndarray,
                     verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compensate multiple patterns in batch.
    
    Args:
        model: Trained CNN model
        target_patterns: Array of target patterns (N, 64, 64, 1)
        verbose: Whether to print progress
        
    Returns:
        Tuple of (predicted_phases, inference_times)
    """
    n_samples = len(target_patterns)
    predicted_phases = []
    inference_times = []
    
    for i in range(n_samples):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i+1}/{n_samples}...")
        
        pattern = target_patterns[i]
        phases, t = compensate_pattern(model, pattern, measure_time=True)
        
        predicted_phases.append(phases)
        inference_times.append(t)
    
    return np.array(predicted_phases), np.array(inference_times)


def evaluate_compensation(model: keras.Model,
                          original_patterns: np.ndarray,
                          faulty_patterns: np.ndarray,
                          original_phases: np.ndarray,
                          verbose: bool = True) -> dict:
    """
    Evaluate pattern compensation performance.
    
    Args:
        model: Trained CNN model
        original_patterns: Original intact array patterns (N, 64, 64, 1)
        faulty_patterns: Faulty array patterns (N, 64, 64, 1)
        original_phases: Original phase configurations (N, 14)
        verbose: Whether to print progress
        
    Returns:
        Dictionary with compensation metrics
    """
    from sklearn.metrics import mean_squared_error
    
    n_samples = len(original_patterns)
    
    # Predict phases for all patterns
    if verbose:
        print(f"Compensating {n_samples} patterns...")
    
    predicted_phases, inference_times = batch_compensate(
        model, original_patterns, verbose=verbose
    )
    
    # Compute phase prediction accuracy
    phase_mae = np.mean(np.abs(predicted_phases - original_phases))
    phase_rmse = np.sqrt(mean_squared_error(original_phases, predicted_phases))
    
    # Compute pattern errors (assuming patterns are in dB scale)
    # Note: In practice, you would need to simulate the recovered patterns
    # using the predicted phases in CST or with an array factor model
    
    results = {
        'n_samples': n_samples,
        'phase_mae_rad': phase_mae,
        'phase_mae_deg': np.rad2deg(phase_mae),
        'phase_rmse_rad': phase_rmse,
        'phase_rmse_deg': np.rad2deg(phase_rmse),
        'predicted_phases': predicted_phases,
        'original_phases': original_phases,
        'inference_times': inference_times,
        'mean_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
    }
    
    return results


def print_compensation_results(results: dict):
    """
    Print compensation evaluation results.
    
    Args:
        results: Dictionary from evaluate_compensation
    """
    print("\n" + "="*60)
    print("Pattern Compensation Results")
    print("="*60)
    print(f"Number of test cases: {results['n_samples']}")
    print(f"\nPhase Prediction Accuracy:")
    print(f"  MAE:  {results['phase_mae_rad']:.6f} rad ({results['phase_mae_deg']:.4f} deg)")
    print(f"  RMSE: {results['phase_rmse_rad']:.6f} rad ({results['phase_rmse_deg']:.4f} deg)")
    print(f"\nInference Time:")
    print(f"  Mean: {results['mean_inference_time_ms']:.2f} ms")
    print(f"  Std:  {results['std_inference_time_ms']:.2f} ms")


def visualize_compensation_results(results: dict, output_dir: str):
    """
    Visualize compensation results.
    
    Args:
        results: Dictionary from evaluate_compensation
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot inference time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['inference_times'] * 1000, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(results['mean_inference_time_ms'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {results['mean_inference_time_ms']:.2f} ms")
    plt.xlabel('Inference Time (ms)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Inference Times', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'inference_time_distribution.png'), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # Plot phase prediction errors
    phase_errors = np.abs(results['predicted_phases'] - results['original_phases'])
    phase_errors_deg = np.rad2deg(phase_errors)
    
    plt.figure(figsize=(12, 6))
    plt.boxplot([phase_errors_deg[:, i] for i in range(14)], 
                labels=[f'E{i+1}' for i in range(14)])
    plt.xlabel('Element', fontsize=12)
    plt.ylabel('Absolute Error (degrees)', fontsize=12)
    plt.title('Phase Prediction Error Distribution per Element', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_error_boxplot.png'), 
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Compensation visualizations saved to: {output_dir}")


def compare_with_optimization_methods():
    """
    Print comparison with traditional optimization methods (Table 4 in paper).
    """
    print("\n" + "="*60)
    print("Computational Time Comparison")
    print("="*60)
    print(f"{'Method':<40} {'Time':<15} {'Speedup':<10}")
    print("-"*60)
    print(f"{'Genetic Algorithm (GA)':<40} {'180-300 s':<15} {'1x':<10}")
    print(f"{'Particle Swarm Optimization (PSO)':<40} {'120-240 s':<15} {'1.5x':<10}")
    print(f"{'CNN Inference (Proposed)':<40} {'~200 ms':<15} {'600-1500x':<10}")
    print("-"*60)


def main():
    """
    Main inference function.
    """
    print("="*60)
    print("CNN Inference for PAA Pattern Compensation")
    print("="*60)
    
    # Load model
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.OUTPUT_DIR, "final_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    model = load_model(model_path)
    
    # Example: Single pattern compensation
    print("\n" + "-"*60)
    print("Example: Single Pattern Compensation")
    print("-"*60)
    
    # Create a dummy pattern for demonstration
    # In practice, this would be loaded from CST or measurement
    dummy_pattern = np.random.rand(64, 64) * 0.5 + 0.25  # Random pattern in [0.25, 0.75]
    
    print("Compensating pattern...")
    predicted_phases, inference_time = compensate_pattern(model, dummy_pattern)
    
    print(f"\nPredicted phases (radians):")
    for i, phase in enumerate(predicted_phases):
        print(f"  Element {i+1:2d}: {phase:.4f} rad ({np.rad2deg(phase):.2f} deg)")
    
    print(f"\nInference time: {inference_time*1000:.2f} ms")
    
    # Reconstruct full 4x4 phase matrix
    full_phases = reconstruct_full_phases(predicted_phases)
    print(f"\nFull 4x4 phase matrix (degrees):")
    print(np.rad2deg(full_phases).round(2))
    
    # Compare with traditional methods
    compare_with_optimization_methods()
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)


if __name__ == "__main__":
    main()
