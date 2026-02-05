import numpy as np

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, X):
        self.input = X
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        # We include learning_rate in the signature even if we don't use it 
        # so it matches the Layer class call in your Network loop.
        
        # np.multiply is the Hadamard product (element-wise)
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Sigmoid(Activation): # Inherit from Activation
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        # Correct way to call the parent constructor
        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            # Derivation: 1 - tanh^2(x)
            return 1 - np.tanh(x)**2

        super().__init__(tanh, tanh_prime)