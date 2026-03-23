"""
Main Script for CNN-based Phased Array Antenna Compensation

This script provides a unified interface for:
1. Data loading and preprocessing
2. Model training
3. Model evaluation
4. Inference and compensation

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode inference --pattern path/to/pattern.npy
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

import config
from data_loader import load_dataset, split_dataset, save_dataset_info
from model import create_cnn_model, compile_model, get_model_summary, count_parameters
from train import train_model
from evaluate import evaluate_model, print_metrics, save_evaluation_results
from inference import load_model, compensate_pattern, compare_with_optimization_methods


def setup_gpu():
    """Configure GPU settings."""
    if config.USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU available: {len(gpus)} device(s)")
                for gpu in gpus:
                    print(f"  - {gpu}")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU available, using CPU")
    else:
        print("GPU disabled in configuration")


def check_data_directory():
    """Check if data directory exists and contains files."""
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Data directory not found: {config.DATA_DIR}")
        print("Please create the directory and add your CST output files.")
        print("Expected format: phase(0,0)=0.phase(0,1)=X... .txt")
        return False
    
    # Check for data files
    import glob
    files = glob.glob(os.path.join(config.DATA_DIR, "phase*.txt"))
    if len(files) == 0:
        print(f"Error: No data files found in {config.DATA_DIR}")
        print("Expected files: phase(0,0)=0.phase(0,1)=X... .txt")
        return False
    
    print(f"Found {len(files)} data files")
    return True


def mode_train(args):
    """Training mode."""
    print("="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Check data
    if not check_data_directory():
        return 1
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    patterns, phases, filenames = load_dataset(config.DATA_DIR)
    
    if len(patterns) == 0:
        print("Error: No data loaded.")
        return 1
    
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
    
    return 0


def mode_evaluate(args):
    """Evaluation mode."""
    print("="*60)
    print("EVALUATION MODE")
    print("="*60)
    
    # Check data
    if not check_data_directory():
        return 1
    
    # Load dataset
    print("\nLoading dataset...")
    patterns, phases, filenames = load_dataset(config.DATA_DIR)
    
    if len(patterns) == 0:
        print("Error: No data loaded.")
        return 1
    
    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        patterns, phases,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        random_seed=config.RANDOM_SEED
    )
    
    # Load model
    model_path = args.model if args.model else os.path.join(config.OUTPUT_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.OUTPUT_DIR, "final_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first or specify a model path with --model")
        return 1
    
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Evaluate
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    
    # Save results
    results_dir = os.path.join(config.OUTPUT_DIR, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    save_evaluation_results(metrics, os.path.join(results_dir, "test_metrics.txt"))
    
    # Generate plots
    print("\nGenerating plots...")
    from evaluate import plot_per_element_accuracy, plot_training_curves, plot_overfitting_analysis
    
    plot_per_element_accuracy(
        metrics['y_test'], metrics['y_pred'],
        os.path.join(results_dir, "per_element_accuracy.png")
    )
    
    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path, os.path.join(results_dir, "training_curves.png"))
        plot_overfitting_analysis(history_path, os.path.join(results_dir, "overfitting_analysis.png"))
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)
    
    return 0


def mode_inference(args):
    """Inference mode."""
    print("="*60)
    print("INFERENCE MODE")
    print("="*60)
    
    # Load model
    model_path = args.model if args.model else os.path.join(config.OUTPUT_DIR, "best_model.h5")
    if not os.path.exists(model_path):
        model_path = os.path.join(config.OUTPUT_DIR, "final_model.h5")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first or specify a model path with --model")
        return 1
    
    model = load_model(model_path)
    
    # Load pattern if provided
    if args.pattern:
        if not os.path.exists(args.pattern):
            print(f"Error: Pattern file not found: {args.pattern}")
            return 1
        
        pattern = np.load(args.pattern)
        print(f"Loaded pattern from: {args.pattern}")
        print(f"Pattern shape: {pattern.shape}")
        
        # Compensate
        print("\nCompensating pattern...")
        predicted_phases, inference_time = compensate_pattern(model, pattern)
        
        print(f"\nPredicted phases (radians):")
        for i, phase in enumerate(predicted_phases):
            print(f"  Element {i+1:2d}: {phase:.4f} rad ({np.rad2deg(phase):.2f} deg)")
        
        print(f"\nInference time: {inference_time*1000:.2f} ms")
        
        # Save results
        if args.output:
            from utils import save_phases_to_file
            save_phases_to_file(predicted_phases, args.output, format='degrees')
            print(f"\nPhases saved to: {args.output}")
    else:
        # Demo mode with random pattern
        print("\nDemo mode: Using random pattern")
        dummy_pattern = np.random.rand(64, 64) * 0.5 + 0.25
        
        predicted_phases, inference_time = compensate_pattern(model, dummy_pattern)
        
        print(f"\nPredicted phases (radians):")
        for i, phase in enumerate(predicted_phases):
            print(f"  Element {i+1:2d}: {phase:.4f} rad ({np.rad2deg(phase):.2f} deg)")
        
        print(f"\nInference time: {inference_time*1000:.2f} ms")
    
    # Compare with traditional methods
    compare_with_optimization_methods()
    
    print("\n" + "="*60)
    print("Inference completed!")
    print("="*60)
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CNN-based Phased Array Antenna Compensation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py --mode train
  
  # Evaluate trained model
  python main.py --mode evaluate
  
  # Run inference on a pattern
  python main.py --mode inference --pattern path/to/pattern.npy
  
  # Use specific model for inference
  python main.py --mode inference --model path/to/model.h5 --pattern path/to/pattern.npy
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'inference'],
                        help='Operation mode')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (for evaluate/inference modes)')
    parser.add_argument('--pattern', type=str, default=None,
                        help='Path to pattern file (for inference mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (for inference mode)')
    
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Run selected mode
    if args.mode == 'train':
        return mode_train(args)
    elif args.mode == 'evaluate':
        return mode_evaluate(args)
    elif args.mode == 'inference':
        return mode_inference(args)
    else:
        print(f"Error: Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
