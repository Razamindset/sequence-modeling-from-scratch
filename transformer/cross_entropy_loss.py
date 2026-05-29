import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        """
        logits: (seq_len, vocab_size) raw scores from LMHead
        targets: (seq_len,) integer token IDs representing ground truth
        """
        self.targets = targets
        N = logits.shape[0]

        # 1. Numerically stable softmax
        shift_logits = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shift_logits)
        self.probs = exps / np.sum(exps, axis=-1, keepdims=True)

        # 2. Compute Negative Log-Likelihood loss
        core_probs = self.probs[np.arange(N), targets]
        loss = -np.mean(np.log(core_probs + 1e-15))
        return loss

    def backward(self):
        """
        The analytical derivative of Softmax + Cross Entropy simplifies beautifully:
        dL/dLogits = (probabilities - target_one_hot) / N
        """
        N = self.probs.shape[0]
        dLogits = self.probs.copy()
        
        # Subtract 1 from the correct token index positions
        dLogits[np.arange(N), self.targets] -= 1.0
        
        # Average over the sequence length 
        return dLogits / N