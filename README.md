# Handwritten Digit Prediction Using ANN

## Project Overview
This project implements a simple Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset. It demonstrates a complete machine learning pipeline, including data loading, preprocessing, model building, training, evaluation, and visualization. The codebase is modularized for clarity and maintainability, following industry-standard practices suitable for a junior machine learning engineer’s portfolio.

### Key Features
- **Dataset**: Uses the MNIST dataset (70,000 grayscale images of 28x28 pixels, 10 digit classes).
- **Model**: A Sequential ANN with a Flatten layer, one hidden Dense layer (32 neurons, ReLU), and a softmax output layer.
- **Technologies**: TensorFlow/Keras, NumPy, Pandas, Matplotlib, Scikit-learn.
- **Tooling**: Supports both `uv` (modern, fast Python package manager) and `pip` for dependency management.
- **Overfitting Mitigation**: Includes a Dropout layer (0.2) to reduce observed overfitting.
- **Outputs**: Generates training/validation loss and accuracy plots (`loss_plot.png`, `accuracy_plot.png`).

This project is hosted on GitHub and can be run locally or in Google Colab, making it accessible for demonstration.

## Repository Structure
```
handwritten-digit-prediction/
├── README.md                # Project documentation
├── requirements.txt         # Dependencies for pip users
├── pyproject.toml          # Dependencies and config for uv users
├── uv.lock                 # Lockfile for reproducible uv installs
├── main.py                 # Entry point to run the project
├── src/
│   ├── __init__.py         # Package initializer
│   ├── data_loader.py      # Loads and preprocesses MNIST data
│   ├── model_builder.py    # Defines the ANN architecture
│   ├── trainer.py          # Handles model training
│   └── evaluator.py        # Performs evaluation and plotting
└── .gitignore              # Ignores unnecessary files (e.g., .venv, *.png)
```

## Prerequisites
- **Python**: Version 3.9–3.12 (3.12 recommended for compatibility with TensorFlow 2.17.0).
- **Operating System**: Tested on Windows; compatible with macOS/Linux.
- **Optional (for GPU)**: NVIDIA GPU, CUDA, and cuDNN for faster TensorFlow training (CPU works fine otherwise).

## Installation

### Option 1: Using uv (Recommended - Faster, Modern)
[uv](https://docs.astral.sh/uv/) is a high-performance Python package manager written in Rust, offering faster installs and reproducible builds via `uv.lock`.

1. **Install uv**:
   - Windows (PowerShell as Admin):
     ```
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   - macOS/Linux:
     ```
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - Verify: `uv --version`

2. **Clone the Repository**:
   ```
   git clone https://github.com/SaifSiddique009/Handwritten-Digit-Prediction-Using-ANN.git
   cd handwritten-digit-prediction
   ```

3. **Set Up Virtual Environment and Install Dependencies**:
   ```
   uv venv --python 3.12  # Creates .venv with Python 3.12
   uv sync                # Installs dependencies from pyproject.toml
   uv lock                # Generates uv.lock for reproducibility
   ```

### Option 2: Using pip (Traditional)
For users preferring the standard Python package manager.

1. **Clone the Repository**:
   ```
   git clone https://github.com/SaifSiddique009/Handwritten-Digit-Prediction-Using-ANN.git
   cd handwritten-digit-prediction
   ```

2. **Create and Activate Virtual Environment**:
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

**Note**: uv users benefit from faster installs and exact version pinning via `uv.lock`. pip users rely on `requirements.txt`, which can be regenerated with `uv export --format requirements-txt > requirements.txt`.

## Usage
Run the project to train the ANN, evaluate on test data, and generate plots:
```
# Using uv
uv run python main.py

# Using pip (with venv activated)
python main.py
```

### Outputs
- **Console**: Displays dataset shapes, model summary, training progress, and test accuracy (~95–97%).
- **Files**: Saves `loss_plot.png` and `accuracy_plot.png` for training/validation loss and accuracy.

## Running in Google Colab
You can run this project in Google Colab for easy demonstration (uses pip, as uv is not natively supported).

1. Open a new notebook: [Google Colab](https://colab.research.google.com).
2. Clone and set up:
   ```bash
   !git clone https://github.com/SaifSiddique009/Handwritten-Digit-Prediction-Using-ANN.git
   %cd handwritten-digit-prediction
   !pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   !python main.py
   ```
4. View plots in the Files tab (`/content/handwritten-digit-prediction/*.png`).
5. Optional: Display plots inline by adding `%matplotlib inline` before running or modifying `src/evaluator.py` to use `plt.show()`.

**Colab Tips**:
- Enable GPU for faster training: Runtime > Change runtime type > GPU.
- If dependency conflicts occur, Colab may restart the runtime (normal).

## Results
- **Test Accuracy**: ~95–97% on MNIST test set after 15 epochs.
- **Plots**: Visualize training/validation loss and accuracy to assess model performance.
- **Overfitting Observation**: Slight overfitting was observed in initial runs. A Dropout layer (0.2) was added to improve generalization.

## Future Improvements
- **Regularization**: Add L2 regularization or increase Dropout rate for better overfitting control.
- **Hyperparameter Tuning**: Use GridSearchCV or Keras Tuner to optimize layer sizes, learning rate, etc.
- **Testing**: Add unit tests with pytest for data loading and model evaluation.
- **Deployment**: Containerize with Docker for reproducible environments.
- **Model Enhancements**: Experiment with CNNs (e.g., Conv2D layers) for higher accuracy.
