import numpy as np

class LSTMLayer:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

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

        for t in range(T):
            xt = x_seq[t:t+1]
            prev_h = self.h_states[t-1]
            concat = np.hstack((xt, prev_h))

            # First compute the big projection
            z = np.dot(concat, self.W) + self.b

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

    def backward(self, dh_out):

        T, H, D = len(dh_out), self.hidden_dim, self.input_dim

        # The things we want to update
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # define the things we want to return
        # 1. dL/dht-1
        dx_seq = np.zeros((T, D))

        # THe things we need during bptt
        dh_next = np.zeros((1, self.hidden_dim))
        dc_next = np.zeros((1, self.hidden_dim))

        for t in reversed(range(T)):
            # get the vlaues from the cache
            i, f, o, g, concat, tanh_ct = self.cache[t]

            # hidden vell highway
            # Total hidden error = (from layer above) + (from next time step)
            dh_total = dh_out[t:t+1] + dh_next

            # 1. To output
            # dL/dO = dL/ht * ht/dO
            do = dh_total * tanh_ct

            # 2. To the Cell State (internal to this time step):
            # dL/dc = dL/dht * dht/dct
            dc_internal = dh_total * o * (1 - tanh_ct** 2)

            # 3. toatal dL/dct = dc_internal + dc_next
            dc_total = dc_internal + dc_next

            # --- THE GATES BLAME GAME ---
            # Based on the forward math: c_t = f*c_{t-1} + i*g
            df = dc_total * self.c_states[t-1]
            di = dc_total * g
            dg = dc_total * i

            # folowing chain rule we have the sigmoid activation too\
            # example
            # f = sigmoid(z)
            # dL/df = sigmoid_prime * dz
            # sigmoid_prime = sigmoid * (1 - sigmoid)
            di_gate = di * (i * (1 - i))
            df_gate = df * (f * (1 - f))
            do_gate = do * (o * (1 - o))
            dg_gate = dg * (1 - g ** 2)

            # --- THE BIG RE-JOIN ---
            # Stack gate gradients back into a (1, 4H) vector to match W
            dz = np.hstack((di_gate, df_gate, do_gate, dg_gate))

            # update the weights
            self.dW += np.dot(concat.T, dz)
            self.db += dz

            # Compute the stuff that will be sent to the previous layer...

            # dL/dconcat = dL/dz * dz/dconcat
            dconcat = np.dot(dz, self.W.T)

            # Now we split this into: dL/dx and dL/dh
            dx_seq[t] = dconcat[:, :D]
            dh_next = dconcat[:, D:]

            # The error to the previous cell state
            # dL/dc_t-1 = dL/dc * dc/dc_t-1
            dc_next = dc_total * f

        return dx_seq
    
    def update(self):
        # Prevent numbers from exploding
        np.clip(self.dW, -1, 1, out=self.dW)
        np.clip(self.db, -1, 1, out=self.db)
        
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.dbs
        
    