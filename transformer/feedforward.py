import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff=2048):
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
        self.hidden = np.dot(x, self.W1) + self.b1

        self.hidden_post = self.relu(self.hidden)

        # Step 2: Contract back to 512 dimensions
        # (seq_len, 2048) @ (2048, 512) -> (seq_len, 512)
        output = np.dot(self.hidden_post, self.W2) + self.b2

        return output
    
    
    def backward(self, dUpstream, X):
        T, D = dUpstream.shape

        dF = dUpstream

        self.db2 = np.sum(dF, axis=0)

        self.dW2 = np.dot(self.hidden_post.T, dF) 


        dH_post = np.dot(dF, self.W2.T)

        # Through hte activation layer
        dH_pre = np.copy(dH_post)

        dH_pre[self.hidden <= 0] = 0

        self.db1 = np.sum(dH_pre, axis=0)

        self.dW1 = np.dot(X.T, dH_pre)

        dL1 = np.dot(dH_pre, self.W1.T)


        self.b2 -= self.db2
        self.W2 -= self.dW2

        self.b1 -= self.db1
        self.W1 -= self.dW1

        return dL1
    
    def get_params(self):
        return {
            "W1": (self.W1, self.dW1), "b1": (self.b1, self.db1),
            "W2": (self.W2, self.dW2), "b2": (self.b2, self.db2)
        }