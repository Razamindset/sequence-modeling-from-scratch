import numpy as np

class Layer:
    def __init__(self, input_size, hidden_size):
        # The Xavier solution scales the weights based on the number of inputs to keep the variance of the signal 
        # constant across layers.
        limit = np.sqrt(6 / (input_size + hidden_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, hidden_size))
        self.bias = np.zeros((1, hidden_size))

        # Some momentum
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, X):
        # X is the activation from the previous layer
        # X is in m dimensions, m is the nodes in prev layer 
        # We faltten it into 1d vecotr so we can do our calculations
        self.input = X.reshape(1, -1)
        
        # calculate and return the pre activations for the layer
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, delta_curr, learning_rate):
        """
        delta_curr: The error signal delta^(l) from the current layer
        self.input: The activation from the previous layer a^(l-1)
        """
        # Reshape and flatten
        delta_curr = delta_curr.reshape(1, -1)

        # 2. Compute Weight Gradient: grad_W = (a^(l-1))^T * delta^(l)
        # On paper: outer product. In code: np.dot of (input.T, delta)
        weights_gradient = np.dot(self.input.T, delta_curr) # this is the equation 3 from our register

        # 3. Compute Bias Gradient: grad_b = delta^(l)
        biases_gradient = delta_curr

        # 4. Compute the Error for the PREVIOUS layer (to be used by the layer before this)
        # On paper: dL/da(l) was expanded to (W^(l+1))^T * delta^(l+1)
        # In code: we dot delta with W.T to keep it as a row vector for the previous layer
        input_gradient = np.dot(delta_curr, self.weights.T)

        # 5. Update Parameters using the Gradient Descent rule
        # self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * biases_gradient
        
        beta = 0.9 # Typical momentum value
        self.v_w = beta * self.v_w + (1 - beta) * weights_gradient
        self.v_b = beta * self.v_b + (1 - beta) * biases_gradient

        self.weights -= learning_rate * self.v_w
        self.bias -= learning_rate * self.v_b

        return input_gradient
    