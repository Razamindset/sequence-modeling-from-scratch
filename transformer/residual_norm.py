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

        self.lr = 0.001

        # Learnable parameters (initialized to 1s and 0s)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x, sublayer_output):
        # 1. Add (Residual Connection)
        # x is the original input, sublayer_output is the result of Attention
        out = x + sublayer_output

        # 2. Norm (Layer Normalization)
        self.var = self.out.var(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)

        # Norm = Gemma * [ (x - mean) / (varience - epsilon)**1/2 ] + Beta
        self.post_norm = (self.out - self.mean) / self.std
        norm_out = self.gamma * self.post_norm + self.beta

        return norm_out
    

    def backward(self, dUpstream):
        # T = sequence length, D = d_model
        T, D = dUpstream.shape

        self.dBeta = np.sum(dUpstream, axis=0)

        self.dGamma = np.sum(dUpstream*self.post_norm, axis=0)

        # 2. Gradient for the normalized input (d_hat_z)
        dPostNorm = dUpstream * self.gamma # shape = (T, d_model)

        # 3. Gradient for the un-normalized input (dZ)
        # This is the "heavy" LayerNorm gradient equation
        # We need means across the feature dimension (axis -1)

        term1 = dPostNorm
        term2 = np.mean(dPostNorm, axis=-1, keepdims=True)
        term3 = self.post_norm * np.mean(dPostNorm * self.post_norm, axis=-1, keepdims=True)

        dPreNorm = (term1 - term2 - term3) / self.std

        # Now we have the gradients here so we can update them...
        # If using a optimizer then we can hand it iver ot the optimizer 

        # There are only two learnable paramerters here one is the 
        # 1. Scaling Gamma
        # 2. Shiftig Beta

        # Now we can divide the gradiet to the L1 output and the output of the FFN that is F1
        # THe exact equations are present in my notes
        return dPreNorm, dPreNorm
    
    def get_params(self):
        return {"gamma": (self.gamma, self.dGamma), "beta": (self.beta, self.dBeta)}

