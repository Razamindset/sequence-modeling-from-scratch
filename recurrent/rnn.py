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

        
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dbh = np.zeros_like(self.bh)

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

    def get_params(self):
        return {
            "Wxh" : (self.Wxh, self.dWxh),
            "Whh" : (self.Whh, self.dWhh),
            "bh" : (self.bh, self.dbh)
        }