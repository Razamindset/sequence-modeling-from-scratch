import numpy as np

class FeedForward:
    def __int__(self, d_model, d_ff=2048):
        # layer 1: 512->2048
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.b1 = np.zeros(d_ff)

        # Layer 2: 2048 -> 512
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2 / d_ff)
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        # (seq_len, 512) @ (512, 2048) -> (seq_len, 2048)
        # Projection to a higher dimension
        hidden = np.dot(x, self.W1) + self.b1

        # Step 2: Contract back to 512 dimensions
        # (seq_len, 2048) @ (2048, 512) -> (seq_len, 512)
        output = np.dot(hidden, self.W2) + self.b2

        return output
    
    