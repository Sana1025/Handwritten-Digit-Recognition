import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("MNIST Handwritten Digit Recognition - Neural Network Demo")
print("=" * 70)


class NeuralNetwork:
        
    def __init__(self, layer_sizes):
        """
        Initialize the neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases with He initialization
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def softmax(self, z):
        """Softmax activation function"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward pass through the network"""
        activations = [X]
        z_values = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i < self.num_layers - 2:  # Hidden layers use ReLU
                a = self.relu(z)
            else:  # Output layer uses softmax
                a = self.softmax(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def compute_loss(self, y_true, y_pred):
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward_propagation(self, X, y, activations, z_values):
        """Backward pass to compute gradients"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient
        delta = activations[-1].copy()
        delta[range(m), y] -= 1
        delta /= m
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            grad_w = np.dot(activations[i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(z_values[i - 1])
        
        return gradients_w, gradients_b
    
    def update_parameters(self, gradients_w, gradients_b, learning_rate):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32, learning_rate=0.01):
        """Train the neural network"""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(m)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                activations, z_values = self.forward_propagation(X_batch)
                gradients_w, gradients_b = self.backward_propagation(X_batch, y_batch, activations, z_values)
                self.update_parameters(gradients_w, gradients_b, learning_rate)
            
            # Compute training metrics
            train_activations, _ = self.forward_propagation(X_train)
            train_loss = self.compute_loss(y_train, train_activations[-1])
            train_pred = np.argmax(train_activations[-1], axis=1)
            train_acc = np.mean(train_pred == y_train)
            
            # Compute validation metrics
            val_activations, _ = self.forward_propagation(X_val)
            val_loss = self.compute_loss(y_val, val_activations[-1])
            val_pred = np.argmax(val_activations[-1], axis=1)
            val_acc = np.mean(val_pred == y_val)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:2d}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - Val_Loss: {val_loss:.4f} - Val_Acc: {val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

def create_digit_pattern(digit, size=28):
    img = np.zeros((size, size))
    
    if digit == 0:
        # Draw circle
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 3
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        outer_mask = (x - center)**2 + (y - center)**2 <= (radius + 2)**2
        img[outer_mask & ~mask] = 1.0
        
    elif digit == 1:
        # Draw vertical line
        img[:, size//2-1:size//2+2] = 1.0
        img[5:10, size//2-3:size//2+2] = 1.0  # top slant
        
    elif digit == 2:
        # Draw 2 shape
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[5:15, 17:20] = 1.0  # right top
        img[12:25, 8:11] = 1.0  # left bottom
        
    elif digit == 3:
        # Draw 3 shape
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[:, 17:20] = 1.0  # right side
        
    elif digit == 4:
        # Draw 4 shape
        img[:15, 8:11] = 1.0  # left vertical
        img[12:15, 8:20] = 1.0  # horizontal
        img[:, 17:20] = 1.0  # right vertical
        
    elif digit == 5:
        # Draw 5 shape
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[5:15, 8:11] = 1.0  # left top
        img[12:25, 17:20] = 1.0  # right bottom
        
    elif digit == 6:
        # Draw 6 shape
        img[5:25, 8:11] = 1.0  # left side
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[12:25, 17:20] = 1.0  # right bottom
        
    elif digit == 7:
        # Draw 7 shape
        img[5:8, 8:20] = 1.0  # top
        img[:, 17:20] = 1.0  # right diagonal
        
    elif digit == 8:
        # Draw 8 shape
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[:, 8:11] = 1.0  # left
        img[:, 17:20] = 1.0  # right
        
    elif digit == 9:
        # Draw 9 shape
        img[5:15, 17:20] = 1.0  # right side
        img[5:8, 8:20] = 1.0  # top
        img[12:15, 8:20] = 1.0  # middle
        img[22:25, 8:20] = 1.0  # bottom
        img[5:15, 8:11] = 1.0  # left top
        
    return img

def generate_synthetic_mnist(n_samples_per_digit=200, noise_level=0.1):
    X = []
    y = []
    
    for digit in range(10):
        for _ in range(n_samples_per_digit):
            # Create base pattern
            img = create_digit_pattern(digit)
            
            # Add random noise
            noise = np.random.randn(28, 28) * noise_level
            img = img + noise
            
            # Random shift
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            img = np.roll(img, shift_x, axis=1)
            img = np.roll(img, shift_y, axis=0)
            
            # Random rotation (slight)
            if np.random.rand() > 0.5:
                img = np.rot90(img, k=np.random.choice([-1, 1]) * (np.random.rand() < 0.1))
            
            # Clip values
            img = np.clip(img, 0, 1)
            
            # Flatten
            X.append(img.flatten())
            y.append(digit)
    
    return np.array(X), np.array(y)

print("\n1. Generating synthetic MNIST-like dataset...")
print("   (Creating handwritten digit patterns...)")

X, y = generate_synthetic_mnist(n_samples_per_digit=200, noise_level=0.15)

print(f"   Total samples: {X.shape[0]}")
print(f"   Features per sample: {X.shape[1]} (28x28 pixels)")

# Split into train, validation, and test sets
indices = np.random.permutation(len(X))
train_idx = indices[:int(0.7 * len(X))]
val_idx = indices[int(0.7 * len(X)):int(0.85 * len(X))]
test_idx = indices[int(0.85 * len(X)):]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Validation samples: {X_val.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")


print("\n2. Building neural network...")
layer_sizes = [784, 128, 64, 10]  # Input: 784, Hidden: 128, 64, Output: 10
print(f"   Architecture: {' → '.join(map(str, layer_sizes))}")

model = NeuralNetwork(layer_sizes)

total_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
print(f"   Total parameters: {total_params:,}")

print("\n3. Training the model...")
print("   (Training on synthetic data...)\n")

history = model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=30,
    batch_size=32,
    learning_rate=0.1
)

print("\n4. Evaluating model on test set...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"   Correct Predictions: {np.sum(y_pred == y_test):,} / {len(y_test):,}")

print("\n5. Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))


print("\n6. Generating visualizations...")

# Create comprehensive results figure
fig = plt.figure(figsize=(20, 10))

# Plot 1: Training History - Accuracy
ax1 = plt.subplot(2, 5, 1)
epochs_range = range(1, len(history['train_acc']) + 1)
ax1.plot(epochs_range, history['train_acc'], label='Training', linewidth=2.5, 
         color='#2E86AB', marker='o', markersize=4)
ax1.plot(epochs_range, history['val_acc'], label='Validation', linewidth=2.5, 
         color='#A23B72', marker='s', markersize=4)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - Loss
ax2 = plt.subplot(2, 5, 2)
ax2.plot(epochs_range, history['train_loss'], label='Training', linewidth=2.5, 
         color='#2E86AB', marker='o', markersize=4)
ax2.plot(epochs_range, history['val_loss'], label='Validation', linewidth=2.5, 
         color='#A23B72', marker='s', markersize=4)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
ax3 = plt.subplot(2, 5, 3)
cm = confusion_matrix(y_test, y_pred)
im = ax3.imshow(cm, cmap='YlGnBu', aspect='auto')
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=10)
ax3.set_xlabel('Predicted Label', fontsize=11)
ax3.set_ylabel('True Label', fontsize=11)
ax3.set_xticks(range(10))
ax3.set_yticks(range(10))

# Add text annotations
for i in range(10):
    for j in range(10):
        text = ax3.text(j, i, cm[i, j], ha="center", va="center", 
                       color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)

plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# Plot 4: Per-digit Accuracy
ax4 = plt.subplot(2, 5, 4)
digit_accuracy = []
for digit in range(10):
    mask = y_test == digit
    if mask.sum() > 0:
        digit_acc = np.mean(y_pred[mask] == digit)
        digit_accuracy.append(digit_acc * 100)
    else:
        digit_accuracy.append(0)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
bars = ax4.bar(range(10), digit_accuracy, color=colors, edgecolor='black', linewidth=1.5)
ax4.set_title('Accuracy by Digit', fontsize=14, fontweight='bold', pad=10)
ax4.set_xlabel('Digit', fontsize=11)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.set_xticks(range(10))
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, digit_accuracy)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 5: Sample digits from dataset
ax5 = plt.subplot(2, 5, 5)
sample_digits = []
for digit in range(10):
    idx = np.where(y_test == digit)[0]
    if len(idx) > 0:
        sample_digits.append(X_test[idx[0]].reshape(28, 28))

if len(sample_digits) == 10:
    combined = np.hstack(sample_digits)
    ax5.imshow(combined, cmap='gray')
    ax5.set_title('Sample Digits (0-9)', fontsize=14, fontweight='bold', pad=10)
    ax5.axis('off')

# Plots 6-10: Sample Correct Predictions
correct_indices = np.where(y_pred == y_test)[0]
if len(correct_indices) >= 5:
    sample_correct = np.random.choice(correct_indices, 5, replace=False)
    
    for i, idx in enumerate(sample_correct):
        ax = plt.subplot(2, 5, 6 + i)
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        confidence = y_proba[idx][y_pred[idx]] * 100
        ax.set_title(f'✓ Label: {y_test[idx]}\nPred: {y_pred[idx]} ({confidence:.0f}%)', 
                     fontsize=11, color='green', fontweight='bold')
        ax.axis('off')

plt.suptitle('MNIST Digit Recognition - Neural Network Training Results', 
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/mnist_training_results.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: mnist_training_results.png")

# Create figure for misclassified examples
incorrect_indices = np.where(y_pred != y_test)[0]

if len(incorrect_indices) > 0:
    fig2 = plt.figure(figsize=(20, 4))
    n_samples = min(10, len(incorrect_indices))
    sample_incorrect = np.random.choice(incorrect_indices, n_samples, replace=False)
    
    for i, idx in enumerate(sample_incorrect):
        ax = plt.subplot(1, 10, i + 1)
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        confidence = y_proba[idx][y_pred[idx]] * 100
        ax.set_title(f'✗ True: {y_test[idx]}\nPred: {y_pred[idx]}\n({confidence:.0f}%)', 
                     fontsize=10, color='red', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Misclassified Digits', fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/mnist_errors.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: mnist_errors.png")

fig3 = plt.figure(figsize=(14, 8))
ax = plt.subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw layers
layer_positions = [1.5, 4, 6.5, 9]
layer_names = ['Input\nLayer\n(784)', 'Hidden\nLayer 1\n(128)', 'Hidden\nLayer 2\n(64)', 'Output\nLayer\n(10)']
layer_heights = [7, 5, 4, 3]

for i, (pos, name, height) in enumerate(zip(layer_positions, layer_names, layer_heights)):
    # Draw rectangle for layer
    rect = plt.Rectangle((pos - 0.3, 5 - height/2), 0.6, height, 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Add label
    ax.text(pos, 1, name, ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Draw connections to next layer
    if i < len(layer_positions) - 1:
        for j in range(3):
            y_start = 5 + (j - 1) * 0.5
            y_end = 5 + (j - 1) * 0.3
            ax.plot([pos + 0.3, layer_positions[i+1] - 0.3], 
                   [y_start, y_end], 'gray', alpha=0.3, linewidth=0.5)

# Add activation functions
ax.text(2.75, 8.5, 'ReLU', ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(5.25, 8.5, 'ReLU', ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(7.75, 8.5, 'Softmax', ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.title('Neural Network Architecture for MNIST Digit Recognition', 
         fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/mnist_architecture.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: mnist_architecture.png")


print("\n7. Saving the trained model...")
model_data = {
    'weights': model.weights,
    'biases': model.biases,
    'layer_sizes': model.layer_sizes
}
with open('/mnt/user-data/outputs/mnist_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("   ✓ Saved: mnist_model.pkl")


readme_content = """# MNIST Handwritten Digit Recognition

""".format(
    train_samples=len(X_train),
    val_samples=len(X_val),
    test_samples=len(X_test),
    params=total_params,
    accuracy=test_accuracy * 100,
    correct=np.sum(y_pred == y_test),
    total=len(y_test)
)

with open('/mnt/user-data/outputs/README.md', 'w') as f:
    f.write(readme_content)
print("   ✓ Saved: README.md")


print("\n" + "=" * 70)
print("TRAINING COMPLETE - Summary")
print("=" * 70)
print(f"Network Architecture:     {' → '.join(map(str, layer_sizes))}")
print(f"Total Parameters:         {total_params:,}")
print(f"Training Samples:         {X_train.shape[0]:,}")
print(f"Validation Samples:       {X_val.shape[0]:,}")
print(f"Test Samples:             {X_test.shape[0]:,}")
print(f"Final Training Accuracy:  {history['train_acc'][-1] * 100:.2f}%")
print(f"Final Val Accuracy:       {history['val_acc'][-1] * 100:.2f}%")
print(f"Test Accuracy:            {test_accuracy * 100:.2f}%")
print(f"Correct Predictions:      {np.sum(y_pred == y_test):,} / {len(y_test):,}")
print(f"Misclassifications:       {np.sum(y_pred != y_test):,}")
print("=" * 70)

print("\n✓ Neural network trained successfully!")
print("\nGenerated Files:")
print("  • mnist_training_results.png - Training history and predictions")
print("  • mnist_errors.png - Misclassified examples")
print("  • mnist_architecture.png - Network diagram")
print("  • mnist_model.pkl - Trained model")
print("  • README.md - Project documentation")