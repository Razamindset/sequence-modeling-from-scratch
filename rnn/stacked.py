import numpy as np

class RNNLayer:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Weights for this specific layer
        self.Wxh = np.random.randn(input_dim, hidden_dim) * 0.1
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.bh = np.zeros((1, hidden_dim))

    def forward(self, x_seq):
        """
        x_seq: (Time, Input_Dim)
        Returns: (Time, Hidden_Dim) sequence
        """
        T = len(x_seq)
        self.h_states = {-1: np.zeros((1, self.hidden_dim))}
        self.x_seq = x_seq # Save for backward
        
        h_out = np.zeros((T, self.hidden_dim))

        for t in range(T):
            # The classic RNN math
            self.h_states[t] = np.tanh(
                np.dot(x_seq[t:t+1], self.Wxh) + 
                np.dot(self.h_states[t-1], self.Whh) + 
                self.bh
            )
            h_out[t] = self.h_states[t]
            
        return h_out

    def backward(self, dh_out):
        """
        dh_out: Gradient coming from the layer ABOVE (Time, Hidden_Dim) also called dh_out in notes
        Returns: dx_seq (Gradient to send to the layer BELOW)
        """
        T = len(dh_out)

        # the gradient of the loss with respect to the inputs of the current layer.
        # This layer recieved the greads wrt each time stamp and must reutrn the same way for compatibity
        dx_seq = np.zeros((T, self.input_dim))

        # The gradient for this step is added to the over all error at the next layer
        dh_next = np.zeros((1, self.hidden_dim))
        
        # Internal gradient storage
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

        for t in reversed(range(T)):
            # Total gradient = (From Layer Above) + (From Next Time Step)
            dh_total = dh_out[t:t+1] + dh_next
            
            # 1. Tanh Jacobian gate
            db_t = (1 - self.h_states[t]**2) * dh_total
            
            # 2. Collect gradients for weights
            self.dWxh += np.dot(self.x_seq[t:t+1].T, db_t)
            self.dWhh += np.dot(self.h_states[t-1].T, db_t)
            self.dbh += db_t
            
            # 3. Pass error HORIZONTALLY (to t-1)
            # The Horizontal Chain Rule (dh_next) This is the error flowing back to the previous hidden state (h_t-1).
            # The Forward Connection: z_t = ... + h_t-1 * W_hh + ...
            # The Chain Rule: dL/dh_t-1 = (dL/dz_t) * (dz_t/dh_t-1)
            # dL/dz_t is your db_t.
            # dz_t/dh_t-1 is the derivative of the linear term (h_t-1 * W_hh) with respect to h_t-1, which is just W_hh.
            # The Result: dh_next = db_t * transpose(W_hh)
            dh_next = np.dot(db_t, self.Whh.T)
            
            # 4. Pass error VERTICALLY (to layer below)
            # This is the error flowing back to the input from the layer below (x_t).
            # The Forward Connection: z_t = x_t * W_xh + ...
            # The Chain Rule: dL/dx_t = (dL/dz_t) * (dz_t/dx_t) = db_t * Wxh
            dx_seq[t] = np.dot(db_t, self.Wxh.T)
            
        return dx_seq

    def update(self):
        # Clip to prevent exploding gradients
        for g in [self.dWxh, self.dWhh, self.dbh]:
            np.clip(g, -5, 5, out=g)
            
        self.Wxh -= self.lr * self.dWxh
        self.Whh -= self.lr * self.dWhh
        self.bh -= self.lr * self.dbh

class StackedRNNModel:
    def __init__(self, layers_dims, output_dim, lr=0.01):
        # layers_dims = [input_dim, h1_dim, h2_dim, ...]
        self.layers = []
        for i in range(len(layers_dims) - 1):
            self.layers.append(RNNLayer(layers_dims[i], layers_dims[i+1], lr))
            
        # Final Output Head (Dense Layer)
        self.Why = np.random.randn(layers_dims[-1], output_dim) * 0.1
        self.by = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, x):
        # Pass through RNN layers like an MLP
        current_data = x
        for layer in self.layers:
            current_data = layer.forward(current_data)
        
        # Last Hidden -> Softmax
        self.last_h = current_data
        y_raw = np.dot(current_data, self.Why) + self.by
        # Softmax over the last dimension
        exp_y = np.exp(y_raw - np.max(y_raw, axis=1, keepdims=True)) 
        return exp_y / np.sum(exp_y, axis=1, keepdims=True)

    def backward(self, probs, targets):
        # 1. Output Layer Gradient
        T = len(targets)
        dy = probs.copy()
        for t in range(T):
            dy[t, targets[t]] -= 1
        
        dWhy = np.dot(self.last_h.T, dy)
        dby = np.sum(dy, axis=0, keepdims=True)
        
        # 2. Gradient to send into the RNN stack
        dh_from_output = np.dot(dy, self.Why.T)
        
        # 3. Backprop through RNN layers (Reverse Order)
        current_grad = dh_from_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad)
            
        # Update everything
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby
        for layer in self.layers:
            layer.update()


# 1. Setup small dummy data
# 3 time steps, vocab size of 4
vocab_size = 4
sent_inputs = np.eye(vocab_size)[:3] # One-hots for first 3 chars
targets = [1, 2, 3]                  # Target indices for those steps

# 2. Initialize a 2-layer Stacked RNN
# Input(4) -> Layer1(8) -> Layer2(8) -> Output(4)
model = StackedRNNModel(layers_dims=[vocab_size, 8, 8], output_dim=vocab_size, lr=0.1)

print("--- Step 1: Forward Pass ---")
probs = model.forward(sent_inputs)
print(f"Output Shape (Time, Vocab): {probs.shape}") 
# Expected: (3, 4)

print("\n--- Step 2: Backward Pass ---")
# This triggers the dh_out -> dx_seq chain through both layers
model.backward(probs, targets)
print("Backward Pass Successful (No dimension mismatches!)")

print("\n--- Step 3: Parameter Update ---")
# This checks if dWxh, dWhh, etc., were calculated and stored
model.layers[0].update()
print("Update Successful!")