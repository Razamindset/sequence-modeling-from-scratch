import numpy as np

class ADAM:
    def __init__(self, layers, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0  # Timestep for bias correction

        # we need m and v for each weight and bias in each layer
        self.m = []
        self.v = []

        for layer in self.layers:
            # We assume each layer has a .params() method returning (weight, gradient) tuples
            layer_m = {name: np.zeros_like(p) for name, (p, grad) in layer.get_params().items()}
            layer_v = {name: np.zeros_like(p) for name, (p, grad) in layer.get_params().items()}
            self.m.append(layer_m)
            self.v.append(layer_v)

    def step(self):
        self.t += 1

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Change params to params.items() here!
            params = layer.get_params()
            for name, (p, grad) in params.items(): 
                np.clip(grad, -1, 1, out=grad) # Clip in-place
                # 1. Update biased first moment estimate
                self.m[i][name] = self.beta1 * self.m[i][name] + (1 - self.beta1) * grad
                
                # 2. Update biased second raw moment estimate
                self.v[i][name] = self.beta2 * self.v[i][name] + (1 - self.beta2) * (grad**2)
                
                # 3. Bias correction
                m_hat = self.m[i][name] / (1 - self.beta1**self.t)
                v_hat = self.v[i][name] / (1 - self.beta2**self.t)
                
                # 4. Update
                # Use -= to modify the weight array in-place
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)