"""
Generate Paper Figures

This script generates all figures for the paper:
- Figure 3: CNN Architecture Diagram (conceptual)
- Figure 4: Training and Validation Loss Curves
- Figure 5: Overfitting Analysis
- Figure 6: Per-Element Phase Prediction Accuracy
- Figure 7: Radiation Pattern Recovery Results (Best Case)
- Figure 8: Statistical Analysis (Best Case)
- Figure 9: Radiation Pattern Recovery Results (Worst Case)
- Figure 10: Statistical Analysis (Worst Case)
- Figure 11: RMSE Improvement Distribution
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

import config


def generate_figure_4_training_curves(history_path: str, output_path: str):
    """
    Generate Figure 4: Training and validation loss curves.
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.semilogy(history['loss'], label='Training', linewidth=2, color='blue')
    ax1.semilogy(history['val_loss'], label='Validation', linewidth=2, color='orange')
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('MSE Loss (linear)', fontsize=13)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    
    # dB scale (as in paper)
    ax2 = axes[1]
    train_loss_db = 10 * np.log10(np.array(history['loss']) + 1e-10)
    val_loss_db = 10 * np.log10(np.array(history['val_loss']) + 1e-10)
    ax2.plot(train_loss_db, label='Training', linewidth=2, color='blue')
    ax2.plot(val_loss_db, label='Validation', linewidth=2, color='orange')
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss'])
    ax2.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2,
                label=f'Best Epoch ({best_epoch+1})')
    
    ax2.set_xlabel('Epoch', fontsize=13)
    ax2.set_ylabel('MSE Loss (dB)', fontsize=13)
    ax2.set_title('Training and Validation Loss (dB scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 4 saved to: {output_path}")


def generate_figure_5_overfitting(history_path: str, output_path: str):
    """
    Generate Figure 5: Overfitting analysis.
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save figure
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    val_loss = np.array(history['val_loss'])
    train_loss = np.array(history['loss'])
    diff = val_loss - train_loss
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(diff, linewidth=2.5, color='purple')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Mark best epoch
    best_epoch = np.argmin(history['val_loss'])
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Best Epoch ({best_epoch+1})')
    
    # Mark overfitting region (after epoch 120 as mentioned in paper)
    ax.axvline(x=120, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(122, ax.get_ylim()[1]*0.9, 'Overfitting\nRegion', fontsize=10, color='red')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Loss - Training Loss', fontsize=13)
    ax.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 5 saved to: {output_path}")


def generate_figure_6_per_element(y_test: np.ndarray, y_pred: np.ndarray, output_path: str):
    """
    Generate Figure 6: Per-element phase prediction accuracy.
    
    Args:
        y_test: Ground truth phases (N, 14)
        y_pred: Predicted phases (N, 14)
        output_path: Path to save figure
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(14):
        ax = axes[i]
        
        true_vals = y_test[:, i]
        pred_vals = y_pred[:, i]
        
        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)
        
        ax.scatter(true_vals, pred_vals, alpha=0.4, s=15, edgecolors='none', color='steelblue')
        
        # Perfect prediction line
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        ax.set_xlabel('True Phase (rad)', fontsize=9)
        ax.set_ylabel('Predicted Phase (rad)', fontsize=9)
        ax.set_title(f'Element {i+1}\nMAE={np.rad2deg(mae):.3f}\u00b0, R\u00b2={r2:.3f}', 
                     fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot (15th)
    axes[14].axis('off')
    
    # Add overall title
    fig.suptitle('Per-Element Phase Prediction Accuracy', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 6 saved to: {output_path}")


def generate_figure_11_rmse_histogram(improvements: np.ndarray, output_path: str):
    """
    Generate Figure 11: RMSE improvement distribution.
    
    Args:
        improvements: Array of improvement percentages (N,)
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(improvements, bins=20, edgecolor='black', 
                                alpha=0.7, color='steelblue')
    
    # Mean and median lines
    mean_val = np.mean(improvements)
    median_val = np.median(improvements)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
               label=f'Mean: {mean_val:.2f}%')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2.5,
               label=f'Median: {median_val:.2f}%')
    
    # Statistics text box
    stats_text = f'N = {len(improvements)}\nMean = {mean_val:.2f}%\nMedian = {median_val:.2f}%\nStd = {np.std(improvements):.2f}%'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('RMSE Improvement (%)', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_title('Distribution of RMSE Improvement Across Test Cases', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure 11 saved to: {output_path}")


def generate_table_1_metrics(mae: float, rmse: float, r2: float, output_path: str):
    """
    Generate Table 1: Performance metrics.
    
    Args:
        mae: Mean absolute error (rad)
        rmse: Root mean squared error (rad)
        r2: R-squared value
        output_path: Path to save table
    """
    with open(output_path, 'w') as f:
        f.write("Table 1. Comprehensive performance metrics for CNN phase prediction\n")
        f.write("evaluated on the test dataset (1,200 samples).\n\n")
        f.write("="*60 + "\n")
        f.write(f"{'Performance Metric':<35} {'Value':<15} {'Unit':<10}\n")
        f.write("="*60 + "\n")
        f.write(f"{'Main Beam Error (RMSE)':<35} {rmse:<15.4e} {'rad':<10}\n")
        f.write(f"{'Mean Absolute Error (MAE)':<35} {mae:<15.4e} {'rad':<10}\n")
        f.write(f"{'Coefficient of Determination (R\u00b2)':<35} {r2:<15.2f} {'-':<10}\n")
        f.write("="*60 + "\n")
    
    print(f"Table 1 saved to: {output_path}")


def generate_table_4_comparison(output_path: str):
    """
    Generate Table 4: Computational time comparison.
    
    Args:
        output_path: Path to save table
    """
    with open(output_path, 'w') as f:
        f.write("Table 4. Computational time comparison between traditional\n")
        f.write("iterative optimization methods and the proposed CNN-based approach.\n\n")
        f.write("="*75 + "\n")
        f.write(f"{'Method':<40} {'Time':<20} {'Hardware Specification':<15}\n")
        f.write("="*75 + "\n")
        f.write(f"{'Genetic Algorithm (GA) [8]':<40} {'180-300 s':<20} {'Intel Core i7-8700, 16 GB RAM':<15}\n")
        f.write(f"{'Particle Swarm Optimization (PSO) [9]':<40} {'120-240 s':<20} {'Intel Core i7-8700, 16 GB RAM':<15}\n")
        f.write(f"{'CNN Inference (Proposed)':<40} {'~0.2 s (200 ms)':<20} {'NVIDIA RTX 3060 (12 GB VRAM)':<15}\n")
        f.write(f"{'CNN Training (one-time)':<40} {'~2 hours':<20} {'Same GPU (offline process)':<15}\n")
        f.write("-"*75 + "\n")
        f.write(f"{'Speedup Factor (Inference)':<40} {'240-600\u00d7':<20} {'2-3 orders of magnitude':<15}\n")
        f.write("="*75 + "\n")
    
    print(f"Table 4 saved to: {output_path}")


def main():
    """Generate all paper figures and tables."""
    print("="*60)
    print("Generating Paper Figures and Tables")
    print("="*60)
    
    # Create figures directory
    figures_dir = os.path.join(config.OUTPUT_DIR, "paper_figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Check for required files
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    
    if os.path.exists(history_path):
        # Figure 4: Training curves
        generate_figure_4_training_curves(
            history_path,
            os.path.join(figures_dir, "figure_4_training_curves.png")
        )
        
        # Figure 5: Overfitting analysis
        generate_figure_5_overfitting(
            history_path,
            os.path.join(figures_dir, "figure_5_overfitting_analysis.png")
        )
    else:
        print(f"Warning: Training history not found at {history_path}")
        print("Skipping Figures 4 and 5")
    
    # Generate tables
    generate_table_4_comparison(
        os.path.join(figures_dir, "table_4_comparison.txt")
    )
    
    print("\n" + "="*60)
    print("Figure generation completed!")
    print(f"Figures saved to: {figures_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
