import numpy as np

class GRU:
    def __init__(self, input_dim, hidden_dim, lr=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # We need weights for 3 components: Reset (r), Update (z), and Candidate (h_tilde)

        # We will keep different vars for the gates and for the final tildal candidate gate 

        # Weights at gate shape: (input_dim + hidden_dim, 2 * hidden_dim)
        limit_g = np.sqrt(6 / (input_dim + 2 * hidden_dim ))
        self.W_gates = np.random.uniform(-limit_g, limit_g, (input_dim + hidden_dim, 2*hidden_dim))
        self.b_gates = np.zeros((1, 2*hidden_dim))

        # Weight at the candid gate
        limit_candd = np.sqrt(6/input_dim +  hidden_dim)
        self.W_cand = np.random.uniform(-limit_candd, limit_candd, (input_dim + hidden_dim, hidden_dim)) # after r*ht-1 the resulting is the modified hiddden state
        self.b_cand= np.zeros((1, hidden_dim))

        # Inside GRUS.__init__
        self.dW_gates = np.zeros_like(self.W_gates)
        self.db_gates = np.zeros_like(self.b_gates)
        self.dW_cand = np.zeros_like(self.W_cand)
        self.db_cand = np.zeros_like(self.b_cand)
        

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x_seq):
        T, H = len(x_seq), self.hidden_dim

        self.h_states = {-1: np.zeros((1, H))}
        self.cache = {}

        h_out = np.zeros(((T, H)))

        for t in range(T):
            xt = x_seq[t:t+1]

            prev_h = self.h_states[t-1]

            # 1. Calculate Reset and Update gates
            # We project input and previous hidden state

            # xt has shape (1, D) (Input features)
            # prev_h has shape (1, H) (Hidden features)
            # concat results in shape (1, D + H)

            # Why we do this for the gates
            # By adding more columns to the input, we can multiply it by one single weight matrix that handles everything at once.

            # If your W_gates has the shape (D + H, 2H), the multiplication looks like this:
            # (1×(D+H))⋅((D+H)×2H)=(1×2H)

            # This gives you one long row where:
            # The first H columns are the values for the Reset gate.
            # The last H columns are the values for the Update gate.

            concat = np.hstack((xt, prev_h))

            z_all = np.dot(concat, self.W_gates) + self.b_gates

            r = self.sigmoid(z_all[:, :H]) #reset
            z = self.sigmoid(z_all[:, H:2*H]) #update 

            # 2. Calculate Candidate Hidden State
            # Note: Reset gate 'r' modulates the previous hidden state
            concat_reset = np.hstack((xt, r * prev_h))
            h_tilde = np.tanh(np.dot(concat_reset, self.W_cand) + self.b_cand)

            # 3. Final Hidden State
            # h_t = (1 - z) * h_{t-1} + z * h_tilde
            self.h_states[t] = (1 - z) * prev_h + z * h_tilde

            self.cache[t] = (xt, prev_h, r, z, h_tilde, concat, concat_reset)
            h_out[t] = self.h_states[t]

        return h_out
    
    def backward(self, dh_out):
        T, H, D = len(dh_out), self.hidden_dim, self.input_dim
        dh_next = np.zeros((1, H))

        # Initilize the update matrices
        dW_gates = np.zeros_like(self.W_gates)
        db_gates = np.zeros_like(self.b_gates)
        dW_cand = np.zeros_like(self.W_cand)
        db_cand = np.zeros_like(self.b_cand)

        # the error to the input of this layer
        dx_seq = np.zeros((T, D))

        for t in reversed(range(T)):
            xt, prev_h, r, z, h_tilde, concat, concat_reset = self.cache[t]
            dh_total = dh_out[t] + dh_next
            
            # ! Refer to the notes for further explanation of these steps 

            #* 1. We can back drop thorough our multi varied funciton f(z, h_tilde, ht-1)
            dz = dh_total * (h_tilde - prev_h) # step 2 in register
            dh_tilde_lin = dh_total * z # step 1 in register
            dh_prev_mix = dh_total * ( 1 - z) # step 3 in register

            # 2*. Now throught the tanh layer
            dh_tilde_raw = dh_tilde_lin * (1 - h_tilde**2)
            dW_cand += np.dot(concat_reset.T, dh_tilde_raw)
            db_cand += dh_tilde_raw

            #* 3. Gradient thorught the concat reset mess
            d_concat_reset = np.dot(dh_tilde_raw, self.W_cand.T)

            # dx_cand is the input to the tanh tayer that is xt
            dx_cand = d_concat_reset[:, :D] 

            # now dr = d[r*hidden](after the D rows) * ht-1(previous hidden state)
            dr = d_concat_reset[:, D:] * prev_h
            
            # Loss to the previous hidden state
            # d_concat_reset(for ht-1) * d/dht-1(r*ht-1)
            dh_prev_cand = d_concat_reset[:, D:] * r 

            #* 4. Conpute the gradients to the sigmoid layers...
            # here the raw ones correspond to dL/dpreactivation that was represented by a "U" on register
            dr_raw = dr * (r * (1 -r))
            dz_raw = dz * ( z* (1 - z))
            d_gates_raw = np.hstack((dr_raw, dz_raw))

            dW_gates += np.dot(concat.T, d_gates_raw)
            db_gates += d_gates_raw

            # Now let us compute the gradients through the concat
            d_concat = np.dot(d_gates_raw, self.W_gates.T)

            dx_gates = d_concat[:, :D]
            dh_prev_gates = d_concat[:, D:]

            # Let us accumulate the the dh_t and dx_seq
            dx_seq[t] = dx_gates + dx_cand
            dh_next = dh_prev_mix + dh_prev_cand + dh_prev_gates

        return dx_seq

    
    def get_params(self):
        return {
            "W_gates": (self.W_gates, self.dW_gates),
            "b_gates": (self.b_gates, self.db_gates),
            "W_cand":  (self.W_cand, self.dW_cand),
            "b_cand":  (self.b_cand, self.db_cand)
        }