import numpy as np

# Precalculat tthe embeddings 
class PositionalEncoding:
    def __init__(self, d_model, max_len=100):
        self.d_model = d_model

        # 1. Create a matrix of shape (max_len, d_model) filled with zeros
        self.pe = np.zeros((max_len, d_model))

        position = np.arange(max_len).reshape(-1, 1)

        print(position)
        # 3. Calculate the 'div_term' (the denominator 10000^(2i/d_model))

        # x^y = e^{y . ln(x)}
        # e^([0, 2, ..., 500]* (ln(1000)/d_model) )
        # We only need it for half the dimensions because we do sin/cos pairs
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        # 4. Fill the even indices (0, 2, 4...) with sine
        self.pe[:, 0::2] = np.sin(position * div_term)

        # 5. Fill the odd indices (1, 3, 5...) with cosine
        self.pe[:, 1::2] = np.cos(position * div_term)
        
    def forward(self, x_embeddings):
        """
        x_embeddings: Your (seq_len, d_model) matrix from the TokenEmbedding layer
        """
        seq_len = x_embeddings.shape[0]
        
        # Add the positional encoding to the word embeddings
        # We only take the first 'seq_len' rows of our pre-calculated matrix
        print(self.pe[:seq_len, :])
        return x_embeddings + self.pe[:seq_len, :]
    

# --- Simple Test ---
d_model = 4
# Assume we have 3 words from our previous step
word_vectors = np.random.randn(3, d_model) 

pe_layer = PositionalEncoding(d_model=d_model)
final_input = pe_layer.forward(word_vectors)

print("Word Vectors (Semantics):\n", word_vectors)
print("\nFinal Input (Semantics + Position):\n", final_input)