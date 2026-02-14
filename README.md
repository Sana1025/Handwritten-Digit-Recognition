# MNIST Handwritten Digit Recognition

## Project Overview
This project implements a neural network from scratch using only NumPy to recognize handwritten digits (0-9).

## Neural Network Architecture
- **Input Layer**: 784 neurons (28x28 pixel images flattened)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation  
- **Output Layer**: 10 neurons with Softmax activation (one per digit)

## Training Details
- **Optimizer**: Mini-batch Gradient Descent
- **Loss Function**: Cross-Entropy Loss
- **Training Samples**: 1,400
- **Validation Samples**: 300
- **Test Samples**: 300
- **Total Parameters**: 109,386

## Results
- **Test Accuracy**: 97.33%
- **Correct Predictions**: 292 / 300

## Files Generated
1. `mnist_training_results.png` - Training metrics and sample predictions
2. `mnist_errors.png` - Examples of misclassified digits
3. `mnist_architecture.png` - Visual diagram of network architecture
4. `mnist_model.pkl` - Saved trained model

## How It Works
The neural network uses:
- **Forward Propagation**: Passes input through layers to make predictions
- **Backward Propagation**: Computes gradients to update weights
- **ReLU Activation**: Introduces non-linearity in hidden layers
- **Softmax Activation**: Converts outputs to probability distribution

## Usage
To load and use the trained model:

```python
import pickle
import numpy as np

# Load model
with open('mnist_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Make predictions on new data
# predictions = model.predict(X_new)
```

## Key Concepts Demonstrated
- Neural network implementation from scratch
- Forward and backward propagation
- Mini-batch gradient descent
- Activation functions (ReLU, Softmax)
- Cross-entropy loss
- Model evaluation and visualization

Built with NumPy, Matplotlib, and Scikit-learn
