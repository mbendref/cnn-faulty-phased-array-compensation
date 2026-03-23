"""
Evaluation and Visualization Module for CNN-based PAA Compensation

This module provides:
- Model evaluation on test set
- Per-element phase prediction accuracy analysis
- Pattern compensation visualization
- Statistical analysis (CDF, scatter plots)
- Performance metrics computation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Tuple, List, Optional
import json

import config
from data_loader import load_dataset, split_dataset


def load_trained_model(model_path: str) -> keras.Model:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to saved model (.h5 file)
        
    Returns:
        Loaded Keras Model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def evaluate_model(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained Keras Model
        X_test: Test input patterns
        y_test: Test ground truth phases
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Predict on test set
    y_pred = model.predict(X_test, verbose=1)
    
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test.flatten(), y_pred.flatten())
    
    # Per-element metrics
    per_element_mae = np.mean(np.abs(y_test - y_pred), axis=0)
    per_element_rmse = np.sqrt(np.mean((y_test - y_pred)**2, axis=0))
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mae_deg': np.rad2deg(mae),
        'rmse_deg': np.rad2deg(rmse),
        'per_element_mae': per_element_mae,
        'per_element_rmse': per_element_rmse,
        'per_element_mae_deg': np.rad2deg(per_element_mae),
        'per_element_rmse_deg': np.rad2deg(per_element_rmse),
        'y_pred': y_pred,
        'y_test': y_test
    }
    
    return metrics


def print_metrics(metrics: dict):
    """
    Print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics from evaluate_model
    """
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    print(f"Mean Squared Error (MSE):     {metrics['mse']:.8f}")
    print(f"Root Mean Squared Error:      {metrics['rmse']:.6f} rad ({metrics['rmse_deg']:.4f} deg)")
    print(f"Mean Absolute Error:          {metrics['mae']:.6f} rad ({metrics['mae_deg']:.4f} deg)")
    print(f"R-squared (R2):               {metrics['r2']:.4f}")
    
    print("\nPer-Element MAE (degrees):")
    for i, mae_deg in enumerate(metrics['per_element_mae_deg']):
        print(f"  Element {i+1:2d}: {mae_deg:.4f} deg")
    print(f"  Average:   {np.mean(metrics['per_element_mae_deg']):.4f} deg")


def plot_per_element_accuracy(y_test: np.ndarray, y_pred: np.ndarray, 
                               output_path: str, num_elements: int = 14):
    """
    Plot per-element phase prediction accuracy (Figure 6 in paper).
    
    Args:
        y_test: Ground truth phases (N, 14)
        y_pred: Predicted phases (N, 14)
        output_path: Path to save the figure
        num_elements: Number of elements to plot
    """
    # Create subplots for each element
    n_cols = 4
    n_rows = (num_elements + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for i in range(num_elements):
        ax = axes[i]
        
        # Get data for this element
        true_vals = y_test[:, i]
        pred_vals = y_pred[:, i]
        
        # Compute metrics
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        
        # Scatter plot
        ax.scatter(true_vals, pred_vals, alpha=0.5, s=20, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Labels and title
        ax.set_xlabel('True Phase (rad)', fontsize=10)
        ax.set_ylabel('Predicted Phase (rad)', fontsize=10)
        ax.set_title(f'Element {i+1}\nMAE={np.rad2deg(mae):.3f} deg, R\u00b2={r2:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(num_elements, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Per-element accuracy plot saved to: {output_path}")


def plot_training_curves(history_path: str, output_path: str):
    """
    Plot training curves from saved history (Figure 4 in paper).
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save the figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss in linear scale
    ax1 = axes[0]
    ax1.plot(history['loss'], label='Training', linewidth=2, color='blue')
    ax1.plot(history['val_loss'], label='Validation', linewidth=2, color='orange')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Loss in dB scale (as shown in paper)
    ax2 = axes[1]
    train_loss_db = 10 * np.log10(np.array(history['loss']) + 1e-10)
    val_loss_db = 10 * np.log10(np.array(history['val_loss']) + 1e-10)
    ax2.plot(train_loss_db, label='Training', linewidth=2, color='blue')
    ax2.plot(val_loss_db, label='Validation', linewidth=2, color='orange')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MSE Loss (dB)', fontsize=12)
    ax2.set_title('Training and Validation Loss (dB scale)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss'])
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch+1})')
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    ax1.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves plot saved to: {output_path}")


def plot_overfitting_analysis(history_path: str, output_path: str):
    """
    Plot overfitting analysis (Figure 5 in paper).
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save the figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Compute difference
    val_loss = np.array(history['val_loss'])
    train_loss = np.array(history['loss'])
    diff = val_loss - train_loss
    
    plt.figure(figsize=(10, 6))
    plt.plot(diff, linewidth=2, color='purple')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss - Training Loss', fontsize=12)
    plt.title('Overfitting Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation
    best_epoch = np.argmin(history['val_loss'])
    plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
                label=f'Best Epoch ({best_epoch+1})')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Overfitting analysis plot saved to: {output_path}")


def plot_cdf_comparison(original_pattern: np.ndarray, 
                        faulty_pattern: np.ndarray,
                        recovered_pattern: np.ndarray,
                        output_path: str):
    """
    Plot CDF comparison for pattern compensation (Figure 8 in paper).
    
    Args:
        original_pattern: Original intact array pattern (dB)
        faulty_pattern: Faulty array pattern (dB)
        recovered_pattern: CNN-recovered pattern (dB)
        output_path: Path to save the figure
    """
    # Flatten patterns
    orig_flat = original_pattern.flatten()
    faulty_flat = faulty_pattern.flatten()
    rec_flat = recovered_pattern.flatten()
    
    # Compute errors
    faulty_error = np.abs(faulty_flat - orig_flat)
    recovered_error = np.abs(rec_flat - orig_flat)
    
    # Compute CDFs
    faulty_sorted = np.sort(faulty_error)
    recovered_sorted = np.sort(recovered_error)
    
    faulty_cdf = np.arange(1, len(faulty_sorted) + 1) / len(faulty_sorted)
    recovered_cdf = np.arange(1, len(recovered_sorted) + 1) / len(recovered_sorted)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CDF plot
    ax1 = axes[0]
    ax1.plot(faulty_sorted, faulty_cdf, label='Faulty vs Original', 
             linewidth=2, color='orange')
    ax1.plot(recovered_sorted, recovered_cdf, label='Recovered vs Original', 
             linewidth=2, color='blue')
    ax1.set_xlabel('Absolute Error (dB)', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.set_title('CDF of Absolute Error', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Recovered
    ax2 = axes[1]
    ax2.scatter(orig_flat, rec_flat, alpha=0.5, s=10, edgecolors='none')
    min_val = min(orig_flat.min(), rec_flat.min())
    max_val = max(orig_flat.max(), rec_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    rmse_rec = np.sqrt(mean_squared_error(orig_flat, rec_flat))
    ax2.set_xlabel('Original Pattern (dB)', fontsize=12)
    ax2.set_ylabel('Recovered Pattern (dB)', fontsize=12)
    ax2.set_title(f'Original vs Recovered (RMSE={rmse_rec:.4f} dB)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot: Original vs Faulty
    ax3 = axes[2]
    ax3.scatter(orig_flat, faulty_flat, alpha=0.5, s=10, edgecolors='none', color='orange')
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    rmse_faulty = np.sqrt(mean_squared_error(orig_flat, faulty_flat))
    ax3.set_xlabel('Original Pattern (dB)', fontsize=12)
    ax3.set_ylabel('Faulty Pattern (dB)', fontsize=12)
    ax3.set_title(f'Original vs Faulty (RMSE={rmse_faulty:.4f} dB)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"CDF comparison plot saved to: {output_path}")
    print(f"  RMSE (Faulty):   {rmse_faulty:.4f} dB")
    print(f"  RMSE (Recovered): {rmse_rec:.4f} dB")
    print(f"  Improvement: {((rmse_faulty - rmse_rec) / rmse_faulty * 100):.2f}%")


def plot_radiation_pattern(pattern: np.ndarray, title: str, output_path: str,
                           cmap: str = 'jet', vmin: Optional[float] = None,
                           vmax: Optional[float] = None):
    """
    Plot a single radiation pattern (Figures 7, 9 in paper).
    
    Args:
        pattern: 2D radiation pattern array
        title: Plot title
        output_path: Path to save the figure
        cmap: Colormap
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
    """
    plt.figure(figsize=(8, 7))
    
    if vmin is None:
        vmin = pattern.min()
    if vmax is None:
        vmax = pattern.max()
    
    im = plt.imshow(pattern, cmap=cmap, origin='lower', 
                    vmin=vmin, vmax=vmax,
                    extent=[-1, 1, -1, 1])
    
    plt.colorbar(im, label='Gain (dBi)')
    plt.xlabel('u (sinθcosφ)', fontsize=12)
    plt.ylabel('v (sinθsinφ)', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Pattern plot saved to: {output_path}")


def plot_rmse_improvement_histogram(improvements: np.ndarray, output_path: str):
    """
    Plot histogram of RMSE improvement percentages (Figure 11 in paper).
    
    Args:
        improvements: Array of improvement percentages
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(improvements, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(improvements):.2f}%')
    plt.axvline(np.median(improvements), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(improvements):.2f}%')
    
    plt.xlabel('RMSE Improvement (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of RMSE Improvement Across Test Cases', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"RMSE improvement histogram saved to: {output_path}")


def save_evaluation_results(metrics: dict, output_path: str):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Mean Squared Error (MSE):     {metrics['mse']:.8f}\n")
        f.write(f"Root Mean Squared Error:      {metrics['rmse']:.8f} rad\n")
        f.write(f"                              {metrics['rmse_deg']:.4f} degrees\n")
        f.write(f"Mean Absolute Error:          {metrics['mae']:.8f} rad\n")
        f.write(f"                              {metrics['mae_deg']:.4f} degrees\n")
        f.write(f"R-squared (R2):               {metrics['r2']:.4f}\n\n")
        
        f.write("Per-Element Performance (MAE in degrees):\n")
        f.write("-"*60 + "\n")
        for i, mae_deg in enumerate(metrics['per_element_mae_deg']):
            f.write(f"  Element {i+1:2d}: {mae_deg:.4f} deg\n")
        f.write(f"  Average:   {np.mean(metrics['per_element_mae_deg']):.4f} deg\n")


def main():
    """
    Main evaluation function.
    """
    print("="*60)
    print("Model Evaluation for PAA CNN")
    print("="*60)
    
    # Load dataset
    print("\nLoading dataset...")
    patterns, phases, filenames = load_dataset(config.DATA_DIR)
    
    if len(patterns) == 0:
        print("Error: No data loaded.")
        return
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        patterns, phases,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    # Load trained model
    model_path = os.path.join(config.OUTPUT_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.OUTPUT_DIR, "final_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model = load_trained_model(model_path)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    results_dir = os.path.join(config.OUTPUT_DIR, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    save_evaluation_results(metrics, os.path.join(results_dir, "test_metrics.txt"))
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Per-element accuracy (Figure 6)
    plot_per_element_accuracy(
        metrics['y_test'], metrics['y_pred'],
        os.path.join(results_dir, "per_element_accuracy.png")
    )
    
    # Training curves (Figure 4)
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(
            history_path,
            os.path.join(results_dir, "training_curves.png")
        )
        
        # Overfitting analysis (Figure 5)
        plot_overfitting_analysis(
            history_path,
            os.path.join(results_dir, "overfitting_analysis.png")
        )
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
