# src/evaluator.py: Module for evaluating the model and plotting results.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test_scaled, y_test):
    """
    Evaluate the model on test data and print accuracy.
    
    Args:
        model (tensorflow.keras.Sequential): Trained model.
        X_test_scaled (numpy.ndarray): Scaled test images.
        y_test (numpy.ndarray): Test labels.
    """
    y_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_prob, axis=1)  # Get class with highest probability
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

def plot_history(history):
    """
    Plot training/validation loss and accuracy, and save to files.
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history.
    """
    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')
    plt.close()
    
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training vs. Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_plot.png')
    plt.close()
    
    print("Plots saved as 'loss_plot.png' and 'accuracy_plot.png'.")