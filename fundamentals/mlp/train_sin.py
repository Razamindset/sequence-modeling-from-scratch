from Layer import Layer
from losses import mse, mse_prime
from activations import Tanh
from network import Network
import numpy as np
import matplotlib.pyplot as plt

# XOR Gate Data
X_train = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1) 
y_train = np.sin(X_train).reshape(-1, 1)

layers = [
    Layer(1, 20),
    Tanh(),
    Layer(20, 10),
    Tanh(),
    Layer(10, 1)
]

model = Network(layers)

model.train(mse, mse_prime, X_train, y_train, epochs=2000, lr=0.01)

# Create values OUTSIDE the training range (e.g., up to 2*pi)
X_test = np.linspace(-2*np.pi, 2*np.pi, 200).reshape(-1, 1)
y_test = np.sin(X_test)

predictions = model.predict(X_test)
pred_plot = [p[0][0] for p in predictions]

plt.plot(X_test, y_test, label='Real Sine')
plt.plot(X_test, pred_plot, label='NN Prediction')
plt.axvspan(-np.pi, np.pi, alpha=0.2, color='gray', label='Training Range')
plt.legend()
plt.show()