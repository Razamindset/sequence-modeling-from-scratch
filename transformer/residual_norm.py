import numpy as np

# 1. Instead of just taking the output of the Attention layer ($Z$) and moving on,
# we add the original input ($X$) back to it.
# 2. Layer Normalization ("The Leveler") After adding, we normalize the values. Unlike "Batch Norm" (which looks at other sequences in the batch), 
# Layer Norm only looks at the 512 dimensions of the current word. For each word vector,
# we Calculate the mean and variance of its 512 values.
# Shift and scale the values so they have a mean of 0 and a variance of 1
# Apply two learnable parameters, Gamma and Beta 
#  to let the model "re-scale" the data if it needs to.

class ResidualNorm:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        # Learnable parameters (initialized to 1s and 0s)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x, sublayer_output):
        # 1. Add (Residual Connection)
        # x is the original input, sublayer_output is the result of Attention
        out = x + sublayer_output

        # 2. Norm (Layer Normalization)
        mean = out.mean(axis=-1, keepdims=True)
        std = out.std(axis=-1, keepdims=True)

        # Norm = Gemma * [ (x - mean) / (varience - epsilon)**1/2 ] + Beta
        norm_out = self.gamma * (out - mean) / (std + self.eps) + self.beta # root(var) = std

        return norm_out
