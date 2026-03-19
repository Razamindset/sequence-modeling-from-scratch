import numpy as np

class TokenEmbedding:
    def __init__(self, vocab_size, d_model):
        self.d_model = d_model
        # The lookup table: rows = words in vocab, cols = embedding size
        self.weights = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x):
        """
        x is now just a 1D list/array of IDs: [1, 2, 3...]
        """
        # Lookup word vectors
        # Result Shape: (seq_len, d_model)
        embeddings = self.weights[x]
        
        # Scale as per the paper
        return embeddings * np.sqrt(self.d_model)

# --- Simple Test ---
sentence = "Sequence modeling from scratch"
vocab = {"<PAD>": 0, "sequence": 1, "modeling": 2, "from": 3, "scratch": 4}

# 1D input: just the IDs for one sentence
input_ids = [1, 2, 3, 4] 

embed_layer = TokenEmbedding(vocab_size=len(vocab), d_model=4)
output = embed_layer.forward(input_ids)

print("Output Shape (Seq_Len, d_model):", output.shape) 
print("\nEach row is a word's vector:\n", output)