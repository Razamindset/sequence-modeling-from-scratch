import numpy as np

class LMHead:
    def __init__(self, d_model, vocab_size):
        # Weight initialization (Kaiming Normal)
        self.W = np.random.randn(d_model, vocab_size) * np.sqrt(2 / d_model)
        self.b = np.zeros(vocab_size)
        
        # Track gradients for ADAM
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x shape: (seq_len, d_model)
        Returns logits: (seq_len, vocab_size)
        """
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dUpstream):
        """
        dUpstream shape: (seq_len, vocab_size)
        """
        self.dW = np.dot(self.x.T, dUpstream)
        self.db = np.sum(dUpstream, axis=0)
        
        # Gradient flowing back into the Transformer Block
        dX = np.dot(dUpstream, self.W.T)
        return dX

    def get_params(self):
        return {"W": (self.W, self.dW), "b": (self.b, self.db)}