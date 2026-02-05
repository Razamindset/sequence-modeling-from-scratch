from Layer import Layer
from losses import mse, mse_prime
from activations import Sigmoid, Tanh
from network import Network
import numpy as np

# XOR Gate Data
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = np.array([
    [0],
    [1],
    [1],
    [0]
])

layers = [
    Layer(2, 3),
    Tanh(),
    Layer(3, 1),
    Tanh(),
]

model = Network(layers)

model.train(mse, mse_prime, X_train, y_train, epochs=1000, lr=0.1)


print("\n--- XOR Results ---")
print(f"{'Input':<15} | {'Target':<10} | {'Prediction':<15} | {'Error':<10}")
print("-" * 60)

predictions = model.predict(X_train)

for x, y, pred in zip(X_train, y_train, predictions):
    # Extract the scalar from the (1,1) array
    p_val = pred[0][0]
    # p_val = pred[0][0] > 0.5 if 1 else 0
    error = abs(y[0] - p_val)
    print(f"{str(x):<15} | {y[0]:<10} | {p_val:<15.4f} | {error:<10.4f}")