from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from Layer import Layer
from losses import cross_entropy, cross_entropy_prime
from activations import Softmax, Tanh
from network import Network
import numpy as np

# 1. Load and Prepare Data
digits = load_digits()
X = digits.images.reshape(len(digits.images), -1) # Flatten 8x8 to 64 pixels
X = X / 16.0 # Normalize pixels to [0, 1]


# One-hot encode labels (e.g., '3' becomes [0,0,0,1,0,0,0,0,0,0])
y = digits.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y)

# train test split
X_train, X_test, y_train, y_test =  train_test_split(X, y_onehot, train_size=0.8, shuffle=True)

# 2. Define Network for Classification
layers = [
    Layer(64, 32), # 64 Input pixels -> 32 Hidden neurons
    Tanh(),
    Layer(32, 10), # 32 Hidden -> 10 Output classes (0-9)
    Softmax()
]

model = Network(layers)

# 3. Train
# Classification usually needs a smaller learning rate
model.train(cross_entropy, cross_entropy_prime, X_train, y_train, epochs=500, lr=0.01)

# 4. Accuracy Calculation
test_predictions = model.predict(X_test)

# Convert predictions (Softmax arrays) to single digit labels
predicted_labels = np.array([np.argmax(p) for p in test_predictions])

# Convert y_test (One-Hot arrays) back to single digit labels
# axis=1 tells numpy to find the max index for each row
true_labels = np.argmax(y_test, axis=1)

# Now both are 1D arrays of integers, so comparison works perfectly
correct_predictions = np.sum(predicted_labels == true_labels)
accuracy = (correct_predictions / len(y_test)) * 100

print(f"\n--- Test Results ---")
print(f"Total Test Samples: {len(y_test)}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")