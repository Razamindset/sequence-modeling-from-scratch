import numpy as np

class GRUS:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # We need weights for 3 components: Reset (r), Update (z), and Candidate (h_tilde)
        # Weights shape: (input_dim + hidden_dim, 3 * hidden_dim)
        limit = np.sqrt(6 / (input_dim + hidden_dim + hidden_dim))
        self.W = np.random.uniform(-limit, limit, (input+hidden_dim, 3*hidden_dim))
        self.b = np.zeros((1, 3*hidden_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_seq):
        T, H = len(x_seq), self.hidden_dim

        self.h_states = {-1: np.zeros((1, H))}
        self.cache = {}

        h_out = np.zeros(((T, H)))

        for t in range(T):
            xt = x_seq[t:t+1]

            prev_h = self.h_states[-1]

            # 1. Calculate Reset and Update gates
            # We project input and previous hidden state
            concat = np.hstack((xt, prev_h))

            z_all = np.dot(concat, self.W[:, :2*H]) + self.b[:, :2*H]

            r = self.sigmoid(z_all[:, :H]) #reset
            z = self.sigmoid(z_all[:, H:2*H]) #update 

            # 2. Calculate Candidate Hidden State
            # Note: Reset gate 'r' modulates the previous hidden state
            concat_reset = np.hstack((xt, r * prev_h))
            h_tilde = np.tanh(np.dot(concat_reset, self.W[:, 2*H:]) + self.b[:, 2*H:])

            # 3. Final Hidden State
            # h_t = (1 - z) * h_{t-1} + z * h_tilde
            self.h_states[t] = (1 - z) * prev_h + z * h_tilde

            self.cache[t] = (xt, prev_h, r, z, h_tilde, concat, concat_reset)
            h_out[t] = self.h_states[t]

        return h_out
    
    def backward(self):
        pass

    def update(self):
        pass
