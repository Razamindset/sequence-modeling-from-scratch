import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, h, lr=1e-4):
        self.d_model = d_model

        self.h = h
        self.d_k = d_model // h
        self.lr = lr

        self.Wq = np.random.randn(d_model, d_model) / np.sqrt(2/d_model)
        self.Wk = np.random.randn(d_model, d_model) / np.sqrt(2 / d_model)
        self.Wv = np.random.randn(d_model, d_model) / np.sqrt(2 / d_model)

        # Output projection

        self.Wo = np.random.randn(d_model, d_model) * np.sqrt(2 / d_model)

        self.dWq = None
        self.dWk = None
        self.dWv = None
        self.dWo = None

    def softmax(self, x):
        # Subtracting max for numerical stability (prevents exp(large_number) = inf)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, x):
        self.x = x
        T, D = x.shape

        # 1. Linear Projections -> (T, d_model)
        self.q_linear = np.dot(x, self.Wq)
        self.k_linear = np.dot(x, self.Wk)
        self.v_linear = np.dot(x, self.Wv)

        # 2. Split into heads -> (h, T, d_k)
        # Reshape to (T, h, d_k) then transpose to (h, T, d_k)
        self.q = self.q_linear.reshape(T, self.h, self.d_k).transpose(1, 0, 2)
        self.k = self.k_linear.reshape(T, self.h, self.d_k).transpose(1, 0, 2)
        self.v = self.v_linear.reshape(T, self.h, self.d_k).transpose(1, 0, 2)

        # 3. Scaled Dot-Product Attention
        # scores = (h, T, T)
        self.scores = np.matmul(self.q, self.k.transpose(0, 2, 1)) / np.sqrt(self.d_k) 
        self.probs = self.softmax(self.scores)

        # 4. Multiply by V -> (h, T, d_k)
        self.att_out = np.matmul(self.probs, self.v)
        
        # Concat and reshape in to the standard size
        self.concat = self.att_out.transpose(1, 0, 2).reshape(T, self.d_model)

        # Transform by the final output matrix 
        output = np.dot(self.concat, self.Wo) 

        return output

    def backward(self, dUpstream):
        T, D = dUpstream.shape

        # 1. Final projection bvackward
        self.dWo = np.dot(self.concat.T, dUpstream)
        dConcat = np.dot(dUpstream, self.Wo.T) # (T, d_model)

        # Split the grandint into the heads
        # Reshape back to (h, T, d_k)
        dAttnOut = dConcat.reshape(T, self.h, self.d_k).transpose(1, 0, 2)

        # Gradient wrt the value layer
        dv = np.matmul(self.probs.transpose(0, 2, 1), dAttnOut)

        dProbs = np.matmul(dAttnOut, self.v.transpose(0, 2, 1))

        # --- Step 3: Softmax Backward ---
        # Simplified vectorized softmax gradient: S * (dS - sum(dS * S))
        dScores = self.probs * (dProbs - np.sum(dProbs * self.probs, axis=-1, keepdims=True))
        
        # Scaled dot product backward
        dScores_scaled = dScores / np.sqrt(self.d_k)
        
        # dq = dScores_scaled @ k -> (h, T, d_k)
        dq = np.matmul(dScores_scaled, self.k)
        # dk = dScores_scaled.T @ q -> (h, T, d_k)
        dk = np.matmul(dScores_scaled.transpose(0, 2, 1), self.q)

        # --- Step 2: Merge heads back to (T, d_model) ---
        dq_linear = dq.transpose(1, 0, 2).reshape(T, D)
        dk_linear = dk.transpose(1, 0, 2).reshape(T, D)
        dv_linear = dv.transpose(1, 0, 2).reshape(T, D)

        # --- Step 1: Linear Projections Backward ---
        self.dWq = np.dot(self.x.T, dq_linear)
        self.dWk = np.dot(self.x.T, dk_linear)
        self.dWv = np.dot(self.x.T, dv_linear)

        # Gradient flowing back to input X from all 3 branches
        dX = (np.dot(dq_linear, self.Wq.T) + 
              np.dot(dk_linear, self.Wk.T) + 
              np.dot(dv_linear, self.Wv.T))

        return dX
    
    def get_params(self):
        return {
            "Wq": (self.Wq, self.dWq), "Wk": (self.Wk, self.dWk),
            "Wv": (self.Wv, self.dWv), "Wo": (self.Wo, self.dWo)
        }