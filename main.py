# main.py: Entry point for the handwritten digit prediction project.

import sys
from src.data_loader import load_and_preprocess_data
from src.model_builder import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model, plot_history

def main():
    try:
        # Step 1: Load and preprocess data
        print("Loading and preprocessing data...")
        X_train_scaled, y_train, X_test_scaled, y_test = load_and_preprocess_data()
        
        # Step 2: Build the model
        print("Building the model...")
        model = build_model()
        
        # Step 3: Train the model
        print("Training the model...")
        history = train_model(model, X_train_scaled, y_train)
        
        # Step 4: Evaluate and plot
        print("Evaluating the model...")
        evaluate_model(model, X_test_scaled, y_test)
        plot_history(history)
        
        print("Project execution completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()