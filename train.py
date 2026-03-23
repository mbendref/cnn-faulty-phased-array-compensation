"""
Training Script for CNN-based Phased Array Antenna Compensation

This script handles:
- Loading and preprocessing data
- Creating the CNN model
- Training with callbacks (early stopping, LR reduction, checkpointing)
- Saving training history and model weights
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

import config
from data_loader import load_dataset, split_dataset, save_dataset_info
from model import create_cnn_model, compile_model, get_model_summary, count_parameters


def create_callbacks(output_dir: str) -> list:
    """
    Create training callbacks.
    
    Args:
        output_dir: Directory to save checkpoints and logs
        
    Returns:
        List of Keras callbacks
    """
    callback_list = []
    
    # Early stopping
    early_stop = callbacks.EarlyStopping(
        monitor=config.EARLY_STOPPING_MONITOR,
        patience=config.EARLY_STOPPING_PATIENCE,
        mode=config.EARLY_STOPPING_MODE,
        restore_best_weights=True,
        verbose=1
    )
    callback_list.append(early_stop)
    
    # Learning rate reduction on plateau
    lr_reduce = callbacks.ReduceLROnPlateau(
        monitor=config.LR_REDUCTION_MONITOR,
        factor=config.LR_REDUCTION_FACTOR,
        patience=config.LR_REDUCTION_PATIENCE,
        mode=config.LR_REDUCTION_MODE,
        min_lr=config.LR_REDUCTION_MIN_LR,
        verbose=1
    )
    callback_list.append(lr_reduce)
    
    # Model checkpoint
    checkpoint_path = os.path.join(output_dir, "best_model.h5")
    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=config.CHECKPOINT_MONITOR,
        mode=config.CHECKPOINT_MODE,
        save_best_only=config.CHECKPOINT_SAVE_BEST_ONLY,
        verbose=1
    )
    callback_list.append(checkpoint)
    
    # TensorBoard logging (optional)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=False
    )
    callback_list.append(tensorboard)
    
    return callback_list


def plot_training_history(history, output_path: str):
    """
    Plot and save training history.
    
    Args:
        history: Keras History object
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss in dB scale (as shown in paper)
    ax2 = axes[0, 1]
    train_loss_db = 10 * np.log10(np.array(history.history['loss']) + 1e-10)
    val_loss_db = 10 * np.log10(np.array(history.history['val_loss']) + 1e-10)
    ax2.plot(train_loss_db, label='Training Loss', linewidth=2)
    ax2.plot(val_loss_db, label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (dB)')
    ax2.set_title('Training and Validation Loss (dB scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MAE
    ax3 = axes[1, 0]
    ax3.plot(history.history['mae'], label='Training MAE', linewidth=2)
    ax3.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Absolute Error (rad)')
    ax3.set_title('Training and Validation MAE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overfitting analysis (validation - training loss)
    ax4 = axes[1, 1]
    loss_diff = np.array(history.history['val_loss']) - np.array(history.history['loss'])
    ax4.plot(loss_diff, linewidth=2, color='purple')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss - Training Loss')
    ax4.set_title('Overfitting Analysis')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {output_path}")


def plot_learning_rate(history, output_path: str):
    """
    Plot learning rate schedule.
    
    Args:
        history: Keras History object
        output_path: Path to save the plot
    """
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(output_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"Learning rate plot saved to: {output_path}")


def save_training_history(history, output_path: str):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras History object
        output_path: Path to save the JSON file
    """
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
    }
    
    if 'lr' in history.history:
        history_dict['lr'] = [float(x) for x in history.history['lr']]
    
    with open(output_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"Training history saved to: {output_path}")


def train_model(X_train, y_train, X_val, y_val, output_dir: str):
    """
    Train the CNN model.
    
    Args:
        X_train: Training input patterns
        y_train: Training output phases
        X_val: Validation input patterns
        y_val: Validation output phases
        output_dir: Directory to save outputs
        
    Returns:
        Trained model and training history
    """
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    # Create model
    model = create_cnn_model()
    model = compile_model(model)
    
    # Print model summary
    print(get_model_summary(model))
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Create callbacks
    callback_list = create_callbacks(output_dir)
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Record training start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callback_list,
        verbose=1
    )
    
    # Record training end time
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("Training Completed")
    print("="*60)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Best validation MAE: {min(history.history['val_mae']):.6f} rad")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.h5")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    save_training_history(history, os.path.join(output_dir, "training_history.json"))
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))
    plot_learning_rate(history, os.path.join(output_dir, "learning_rate.png"))
    
    # Save training info
    with open(os.path.join(output_dir, "training_info.txt"), 'w') as f:
        f.write("Training Information\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training time: {training_time/60:.2f} minutes\n")
        f.write(f"Total epochs: {len(history.history['loss'])}\n")
        f.write(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}\n")
        f.write(f"Best validation loss: {min(history.history['val_loss']):.8f}\n")
        f.write(f"Best validation MAE: {min(history.history['val_mae']):.8f} rad\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.8f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.8f}\n")
    
    return model, history


def main():
    """
    Main training function.
    """
    print("="*60)
    print("CNN Training for Phased Array Antenna Compensation")
    print("="*60)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    patterns, phases, filenames = load_dataset(config.DATA_DIR)
    
    if len(patterns) == 0:
        print("Error: No data loaded. Please check data directory.")
        return
    
    # Split dataset
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
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, config.OUTPUT_DIR)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.8f}")
    print(f"Test MAE: {test_mae:.6f} rad ({np.rad2deg(test_mae):.4f} deg)")
    print(f"Test RMSE: {np.sqrt(test_mse):.6f} rad ({np.rad2deg(np.sqrt(test_mse)):.4f} deg)")
    
    # Save test results
    with open(os.path.join(config.OUTPUT_DIR, "test_results.txt"), 'w') as f:
        f.write("Test Set Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test Loss (MSE): {test_loss:.8f}\n")
        f.write(f"Test MAE: {test_mae:.8f} rad\n")
        f.write(f"Test MAE: {np.rad2deg(test_mae):.4f} degrees\n")
        f.write(f"Test RMSE: {np.sqrt(test_mse):.8f} rad\n")
        f.write(f"Test RMSE: {np.rad2deg(np.sqrt(test_mse)):.4f} degrees\n")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"All outputs saved to: {config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
