# src/model_builder.py: Module for building the ANN model.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

def build_model():
    """
    Build a simple ANN model for digit classification.
    
    Returns:
        model (tensorflow.keras.Sequential): Compiled Keras model.
    """
    model = Sequential([
        Input(shape=(28, 28)),          # Input layer for 28x28 images
        Flatten(),                      # Flatten 2D image to 1D vector
        Dense(32, activation='relu'),   # Hidden layer with 32 neurons
        Dropout(0.2),                   # Add dropout to reduce overfitting
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer='Adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model