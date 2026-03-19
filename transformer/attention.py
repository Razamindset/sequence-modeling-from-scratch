import numpy as np

class SingleHeadSelfAttention():
    def __init__(self, d_model):

        self.d_model = d_model
        self.scale = np.sqrt(d_model)

        # Xavier initlization
        limit = np.sqrt(6/(2*d_model))
        self.W_q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_k = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_v = np.random.uniform(-limit, limit, (d_model, d_model))

    def softmax(self, x):
        # e(z) = (e^z_i) / sum(all_z_is)
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def forward(self, x):
        # x shape: (seq_len, d_model) -> e.g., (2, 512)

        # 1. Do the linear projections first 
        # (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # 2. Compute self attention scores
        # Dot product of Q and K^T tells us the raw "similarity"
        # (seq_len, d_model) @ (d_model, seq_length) -> (2, 2)
        scores = np.dot(Q, K.T)

        # 3. Scale the scores 
        # This is the "Scaled" part of Scaled Dot-Product Attention
        scaled_scores = scores / self.scale

        # 4. Softmax
        # turn scores into probs
        attention_weights = self.softmax(scaled_scores)

        # 5. Output
        # Multiply weights by V to get the new, context-aware word vectors
        # (2, 2) @ (2, 512) -> (2, 512)
        output = np.dot(attention_weights, V)

        return output, attention_weights
    
    