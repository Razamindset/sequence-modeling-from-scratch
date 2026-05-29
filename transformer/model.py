from mha import MultiHeadAttention
from residual_norm import ResidualNorm
from feedforward import FeedForward

class TransformerBlock:
    def __init__(self, d_model, h, d_ff=2048):
        # Sub-layers
        self.mha = MultiHeadAttention(d_model, h)
        self.norm1 = ResidualNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = ResidualNorm(d_model)
        pass

    def forward(self, x):
        """
        x shape: (seq_len, d_model)
        """
        # 1. Attention Block (MHA + Skip + LayerNorm)
        attn_out = self.mha.forward(x)
        self.x1 = self.norm1.forward(x, attn_out)

        # 2. Feed-Forward Block (FFN + Skip + LayerNorm)
        ffn_out = self.ffn.forward(self.x1)
        x2 = self.norm2.forward(self.x1, ffn_out)

        return x2
    
    def backward(self, dUpstream):
        """
        dUpstream shape: (seq_len, d_model)
        Tracks the exact reverse path of the forward execution.
        """
        # 1. Backprop through LayerNorm 2 (FFN block)
        # Returns: gradient for skip-connection, gradient for FFN output
        dNorm2_skip, dFfn_out = self.norm2.backward(dUpstream)

        # 2. Backprop through FFN
        dFfn_in = self.ffn.backward(dFfn_out)

        # 3. Accumulate gradients at the intermediate highway split (x1)
        dX1 = dNorm2_skip + dFfn_in

        # 4. Backprop through LayerNorm 1 (MHA block)
        # Returns: gradient for input skip-connection, gradient for MHA output
        dNorm1_skip, dMha_out = self.norm1.backward(dX1)

        # 5. Backprop through MHA
        dMha_in = self.mha.backward(dMha_out)

        # 6. Accumulate gradients at the original input highway split (x)
        dX = dNorm1_skip + dMha_in

        return dX

    def get_params(self):
        """
        Flattens all internal parameters from sub-layers into a single 
        dictionary with prefixed keys so ADAM can update them smoothly.
        """
        params = {}
        
        # Pull parameters from all internal layers
        for prefix, layer in [("mha", self.mha), ("norm1", self.norm1), 
                              ("ffn", self.ffn), ("norm2", self.norm2)]:
            for name, (p, grad) in layer.get_params().items():
                params[f"{prefix}_{name}"] = (p, grad)
                
        return params