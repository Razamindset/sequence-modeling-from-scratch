import numpy as np

class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # We need four weights matrices here
        # 1. forget gate
        # 2. Input gate layer
        # 3. Tanh gate
        # 4. Output gate layer used for the final output of the state

        # Instead of creating four differnt matrices we can concatinate this into one matric
        # The input x_t and the prev hidden layer at h_t-1 will first we combined
        # Combined input matric: (1, D+H)
        # Weight matric: (D+H, 4*H)
        # Bias: (1, 4*H)
        self.W = np.random.randn(input_dim+hidden_dim, 4 * hidden_dim) * 0.01
        self.b = np.zeros((1, 4 * hidden_dim))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def forward(self, x_seq):
        """
        x_seq: (Time, Input_Dim) is the x_t
        Returns: (Time, Hidden_Dim) sequence
        """
        T = len(x_seq)
        H = self.hidden_dim

        self.h_states = { -1 : np.zeros((1, H)) }
        self.c_states = { -1 : np.zeros((1, H)) }

        self.cache = {}

        h_out = np.zeros((T, H))

        for t in range(len(T)):
            xt = x_seq[t:t+1]
            prev_h = self.h_states[t-1]
            concat = np.hstack((xt, prev_h))

            # First compute the big projection
            z = np.dot(concat, self.W)

            # slice and perform the activation 
            i = self.sigmoid(z[:, :H])         # Input gate
            f = self.sigmoid(z[:, H:2*H])      # Forget gate
            o = self.sigmoid(z[:, 2*H:3*H])    # Output gate
            g = np.tanh(z[:, 3*H:]) # tanh gate

            # update the cellstate
            self.c_states[t] = f * self.c_states[t-1] + i * g

            # 5. Update the Hidden State
            # Math: h_t = o * tanh(c_t)
            tanh_ct = np.tanh(self.c_states[t])
            self.h_states[t] = o * tanh_ct

            # 6. Save for later (The 'Cache')
            self.cache[t] = (i, f, o, g, concat, tanh_ct)

            h_out[t] = self.h_states[t]

        return h_out

    def backward(self):
        pass

    