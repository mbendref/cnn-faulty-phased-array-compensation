"""
CNN Model Architecture for Phased Array Antenna Pattern Compensation

This module defines the convolutional neural network that learns the mapping
from 2D radiation patterns to excitation phases for the 14 active elements.

Architecture (as described in the paper):
- 4 Convolutional blocks with progressive feature extraction
- Global Average Pooling (GAP) instead of flattening
- 3 Fully connected layers for phase regression
- Total parameters: ~2.8 million
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import config


def create_cnn_model(input_shape: tuple = config.INPUT_SHAPE,
                     output_dim: int = config.NUM_ACTIVE_ELEMENTS,
                     name: str = "PAA_CNN") -> Model:
    """
    Create the CNN model for pattern-to-phase mapping.
    
    Architecture based on the paper:
    - Input: 64x64x1 radiation pattern image
    - 4 Conv blocks with batch norm, ReLU, max pooling, dropout
    - Global Average Pooling
    - 3 Dense layers with dropout
    - Output: 14 phase values (radians)
    
    Args:
        input_shape: Shape of input pattern (H, W, C)
        output_dim: Number of output phase values (14)
        name: Model name
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="input_pattern")
    
    x = inputs
    
    # =========================================================================
    # Block 1: Initial feature extraction
    # Input: 64x64x1 -> Output: 32x32x64
    # =========================================================================
    x = layers.Conv2D(64, kernel_size=(7, 7), padding='same', 
                      name='conv1_1')(x)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.ReLU(name='relu1_1')(x)
    
    x = layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                      name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.ReLU(name='relu1_2')(x)
    
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = layers.Dropout(0.10, name='dropout1')(x)
    
    # =========================================================================
    # Block 2: Feature expansion
    # Input: 32x32x64 -> Output: 16x16x128
    # =========================================================================
    x = layers.Conv2D(128, kernel_size=(5, 5), padding='same',
                      name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.ReLU(name='relu2_1')(x)
    
    x = layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                      name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.ReLU(name='relu2_2')(x)
    
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = layers.Dropout(0.15, name='dropout2')(x)
    
    # =========================================================================
    # Block 3: Deep feature learning
    # Input: 16x16x128 -> Output: 8x8x256
    # =========================================================================
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                      name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.ReLU(name='relu3_1')(x)
    
    x = layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                      name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.ReLU(name='relu3_2')(x)
    
    x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = layers.Dropout(0.20, name='dropout3')(x)
    
    # =========================================================================
    # Block 4: High-level feature refinement
    # Input: 8x8x256 -> Output: 8x8x512
    # =========================================================================
    x = layers.Conv2D(512, kernel_size=(3, 3), padding='same',
                      name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.ReLU(name='relu4_1')(x)
    
    x = layers.Conv2D(512, kernel_size=(3, 3), padding='same',
                      name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.ReLU(name='relu4_2')(x)
    
    # Global Average Pooling instead of flattening
    # Reduces 8x8x512 -> 512
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.30, name='dropout4')(x)
    
    # =========================================================================
    # Fully Connected Layers for Phase Regression
    # =========================================================================
    # Dense layer 1: 512 -> 1024
    x = layers.Dense(1024, name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.ReLU(name='relu_dense1')(x)
    x = layers.Dropout(0.40, name='dropout_dense1')(x)
    
    # Dense layer 2: 1024 -> 256
    x = layers.Dense(256, name='dense2')(x)
    x = layers.ReLU(name='relu_dense2')(x)
    x = layers.Dropout(0.20, name='dropout_dense2')(x)
    
    # Output layer: 256 -> 14 (phase values in radians)
    # Linear activation for regression
    outputs = layers.Dense(output_dim, activation='linear', name='phases')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


def create_dnn_baseline(input_shape: tuple = config.INPUT_SHAPE,
                        output_dim: int = config.NUM_ACTIVE_ELEMENTS,
                        name: str = "PAA_DNN_Baseline") -> Model:
    """
    Create a fully connected DNN baseline for comparison.
    
    This is the baseline model used in the paper for comparison with CNN.
    It has ~52.4 million parameters vs ~2.8 million for the CNN.
    
    Args:
        input_shape: Shape of input pattern
        output_dim: Number of output phase values
        name: Model name
        
    Returns:
        Compiled Keras Model
    """
    inputs = layers.Input(shape=input_shape, name="input_pattern")
    
    # Flatten the input: 64x64x1 = 4096
    x = layers.Flatten(name='flatten')(inputs)
    
    # Dense layers
    x = layers.Dense(2048, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.30, name='dropout1')(x)
    
    x = layers.Dense(1024, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.30, name='dropout2')(x)
    
    x = layers.Dense(512, activation='relu', name='dense3')(x)
    x = layers.Dropout(0.30, name='dropout3')(x)
    
    x = layers.Dense(256, activation='relu', name='dense4')(x)
    x = layers.Dropout(0.20, name='dropout4')(x)
    
    # Output layer
    outputs = layers.Dense(output_dim, activation='linear', name='phases')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=name)
    
    return model


def compile_model(model: Model, 
                  learning_rate: float = config.LEARNING_RATE,
                  loss: str = config.LOSS_FUNCTION,
                  metrics: list = None) -> Model:
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model: Keras Model to compile
        learning_rate: Learning rate for optimizer
        loss: Loss function name
        metrics: List of metrics to track
        
    Returns:
        Compiled Model
    """
    if metrics is None:
        metrics = config.METRICS
    
    # Create optimizer
    if config.OPTIMIZER.lower() == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif config.OPTIMIZER.lower() == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    elif config.OPTIMIZER.lower() == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model


def get_model_summary(model: Model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras Model
        
    Returns:
        Model summary string
    """
    import io
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary = stream.getvalue()
    stream.close()
    return summary


def count_parameters(model: Model) -> int:
    """
    Count trainable parameters in the model.
    
    Args:
        model: Keras Model
        
    Returns:
        Number of trainable parameters
    """
    return model.count_params()


if __name__ == "__main__":
    # Test model creation
    print("Creating CNN model...")
    cnn_model = create_cnn_model()
    cnn_model = compile_model(cnn_model)
    
    print("\n" + "="*60)
    print("CNN Model Summary")
    print("="*60)
    cnn_model.summary()
    
    print(f"\nTotal trainable parameters: {count_parameters(cnn_model):,}")
    
    # Test with dummy input
    import numpy as np
    dummy_input = np.random.randn(2, 64, 64, 1).astype(np.float32)
    dummy_output = cnn_model.predict(dummy_input)
    print(f"\nDummy input shape:  {dummy_input.shape}")
    print(f"Dummy output shape: {dummy_output.shape}")
    print(f"Output range: [{dummy_output.min():.4f}, {dummy_output.max():.4f}]")
    
    # Compare with DNN baseline
    print("\n" + "="*60)
    print("DNN Baseline Model Summary")
    print("="*60)
    dnn_model = create_dnn_baseline()
    dnn_model = compile_model(dnn_model)
    dnn_model.summary()
    print(f"\nTotal trainable parameters: {count_parameters(dnn_model):,}")
