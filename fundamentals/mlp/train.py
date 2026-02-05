from Layer import Layer
from losses import mse, mse_prime
from activations import Sigmoid, Tanh
import numpy as np

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def predict(self, X):
        """
        predict the outputs for n number of inputs 
        
        :param X: all the test inputs (vector)
        """
        results = []

        for x in X:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            
            results.append(output)

        return results
    
    def train(self, loss_func, loss_prime, X_train, y_train, epochs=100, lr=0.001):
        for epoch in range(epochs):
            error = 0

            for x, y in zip(X_train, y_train):

                # perform the forward pass
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                
                error += loss_func(y, output)

                # loss will the error say E
                # dE / d_activations = loss prime 
                grad = loss_prime(y, output)

                for layer in reversed(self.layers):
                    # take loss form each layer and pass it to the previous one 
                    grad = layer.backward(grad, lr)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}  error={error/len(X_train)}')

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