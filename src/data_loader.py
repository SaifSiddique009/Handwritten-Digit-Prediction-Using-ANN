# src/data_loader.py: Module for loading and preprocessing the MNIST dataset.

import tensorflow as tf

def load_and_preprocess_data():
    """
    Load the MNIST dataset and preprocess it by scaling pixel values to [0, 1].
    
    Returns:
        X_train_scaled (numpy.ndarray): Scaled training images.
        y_train (numpy.ndarray): Training labels.
        X_test_scaled (numpy.ndarray): Scaled test images.
        y_test (numpy.ndarray): Test labels.
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Scale pixel values to [0, 1]
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    
    return X_train_scaled, y_train, X_test_scaled, y_test