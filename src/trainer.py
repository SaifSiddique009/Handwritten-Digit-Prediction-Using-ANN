# src/trainer.py: Module for training the model.

def train_model(model, X_train_scaled, y_train):
    """
    Train the ANN model on the scaled training data.
    
    Args:
        model (tensorflow.keras.Sequential): The model to train.
        X_train_scaled (numpy.ndarray): Scaled training images.
        y_train (numpy.ndarray): Training labels.
    
    Returns:
        history (tensorflow.keras.callbacks.History): Training history.
    """
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=15,
        batch_size=100,
        validation_split=0.2
    )
    
    return history