import numpy as np

class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.hidden_dim = hidden_dim
        self.lr = lr

        # initilize weights 
        self.Wxh = np.random.randn(input_dim, hidden_dim) * np.sqrt(1. / input_dim)
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.Why = np.random.randn(hidden_dim, output_dim) * np.sqrt(1. / hidden_dim)
        
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, output_dim))

    def forward(self, x_sequence):
        """
        x_sequence: list of one-hot vectors or (Time, Input_Dim) matrix
        """
        h_states = {-1: np.zeros((1, self.hidden_dim))}
        y_probs = {}

        for t in range(len(x_sequence)):
            x_t = x_sequence[t].reshape(1, -1)

            # h_t = tanh(x_t*Wxh + h_{t-1}*Whh + bh)
            h_states[t] = np.tanh(np.dot(x_t, self.Wxh) + np.dot(h_states[t-1], self.Whh) + self.bh)

            # y_t = softmax(h_t*Why + by)
            y_raw = np.dot(h_states[t], self.Why) + self.by
            y_probs[t] = np.exp(y_raw) / np.sum(np.exp(y_raw))

        return h_states, y_probs

    def backward(self, x_sequence, h_states, y_probs, targets):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        dh_next = np.zeros((1, self.hidden_dim))

        # BPTT: Walking backward through time
        for t in reversed(range(len(x_sequence))):
            # 1. Output error (p - target)
            dy = y_probs[t].copy()
            dy[0, targets[t]] -= 1

            # 2. output weights 
            dWhy += np.dot(h_states[t].T, dy)
            dby += dy

            # Error wrt hidden state 
            dh = np.dot(dy, self.Why.T) + dh_next

            # 4. Error through tanh activation
            db_t = (1 - h_states[t]**2) * dh

            # 5. Compute weights gradeints
            dWxh += np.dot(x_sequence[t].reshape(-1, 1), db_t)
            dWhh += np.dot(h_states[t-1].T, db_t)
            dbh += db_t

            # 6. Pass the error to the previous time step
            dh_next = np.dot(db_t, self.Whh.T)

            # Gradient Clipping (Professional touch to prevent exploding gradients)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return dWxh, dWhh, dWhy, dbh, dby

    
    def update_params(self, grads):
        dWxh, dWhh, dWhy, dbh, dby = grads
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby
