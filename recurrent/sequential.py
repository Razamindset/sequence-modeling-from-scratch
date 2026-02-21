import numpy as np
from optimizers.ADAM import ADAM

class SequentialModel:
    def __init__(self, layers, output_layer, lr=0.01):
        # layers_dims = [input_dim, h1_dim, h2_dim, ...]
        self.layers = layers
        self.output_layer = output_layer
        self.optimizer = ADAM(self.layers + [self.output_layer], learning_rate=lr)

    def forward(self, x):
        # Pass through RNN layers like an MLP
        current_data = x
        for layer in self.layers:
            current_data = layer.forward(current_data)
        
        # Pass the final sequence to the Dense output head
        logits = self.output_layer.forward(current_data)

        # Softmax for probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def train_step(self, x, targets):
        # 1. Forward
        probs = self.forward(x)
        
        # 2. Compute Initial Gradient (Cross-Entropy derivative)
        T = len(targets)
        d_logits = probs.copy()
        for t in range(T):
            d_logits[t, targets[t]] -= 1
        d_logits /= T # Normalize by sequence length
        

        current_grad = self.output_layer.backward(d_logits)
        
        # 3. Backprop through RNN layers (Reverse Order)
        # Store norms for all layers
        
        # 4. Backward through the stack
        for layer in reversed(self.layers):
            # Most layers return (dx_seq, optional_norms)
            res = layer.backward(current_grad)
            current_grad = res[0] if isinstance(res, tuple) else res

        # 5. Optimization step (Adam)
        self.optimizer.step()
        
        # Return loss for monitoring
        loss = -np.sum(np.log(probs[range(T), targets] + 1e-8)) / T

        return loss
