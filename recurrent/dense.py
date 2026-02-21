import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        # Xavier/Glorot Initialization
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros((1, output_dim))
        
        # Initialize gradient placeholders for the ADAM optimizer
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)

    def forward(self, x):
        """
        x: (Time, Input_Dim) from the RNN/GRU layer
        returns: (Time, Output_Dim) logits
        """
        self.input = x 
        # (T, I) dot (I, O) = (T, O)
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, d_logits):
        """
        d_logits: (Time, Output_Dim) gradient from the loss function
        returns: (Time, Input_Dim) gradient to pass back to the RNN/GRU
        """
        # 1. Gradient for weights: (Input_Dim, Time) dot (Time, Output_Dim)
        # This sums the gradients across the entire sequence length
        self.dweights = np.dot(self.input.T, d_logits)
        
        # 2. Gradient for bias: sum across the Time dimension
        self.dbias = np.sum(d_logits, axis=0, keepdims=True)
        
        # 3. Gradient for the layer below (RNN/GRU hidden states)
        # (Time, Output_Dim) dot (Output_Dim, Input_Dim) = (Time, Input_Dim)
        dx = np.dot(d_logits, self.weights.T)
        
        return dx

    def get_params(self):
        """Standard interface for your ADAM optimizer"""
        return {
            "weights": (self.weights, self.dweights),
            "bias": (self.bias, self.dbias)
        }