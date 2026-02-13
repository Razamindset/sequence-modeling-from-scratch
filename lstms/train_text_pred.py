import numpy as np
from main import StackedLSTM 

# This code does not converges very well because we need to do optimizations to the architecute vefore we can truly leverage the benefits of the LSTMS this includes initilization, batches, and learning rate opeimizers and weight decay... I don't really wanna spend time on optimization for now...
# ! and this training code is completely AI generated 

# 1. THE EXTENDED CORPUS
text = """
the quick brown fox jumps over the lazy dog. 
deep learning is a subset of machine learning based on artificial neural networks.
you are building an lstm from scratch in python using numpy for matrix math.
sequence modeling allows us to predict the next element in a series.
memory cells in an lstm help prevent the vanishing gradient problem.
programming is the art of telling a computer what to do.
data science combines math, statistics, and domain expertise.
artificial intelligence is transforming the way we interact with technology.
"""

# Vocabulary setup
chars = sorted(list(set(text)))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
vocab_size = len(chars)

# 2. UTILITY FUNCTIONS
def encode(t):
    return np.array([char_to_ix[c] for c in t])

def one_hot(indices, dim):
    out = np.zeros((len(indices), dim))
    out[range(len(indices)), indices] = 1
    return out

# 3. SOFTMAX HEAD (Separated as requested)
class SoftmaxHead:
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, probs, targets):
        T = targets.shape[0]
        loss = -np.sum(np.log(probs[range(T), targets] + 1e-8)) / T
        dx = probs.copy()
        dx[range(T), targets] -= 1
        return loss, dx / T

# 4. SLIDING WINDOW SAMPLING (Fixes the "Amnesia" without changing LSTM)
def sample(model, head, start_str, length, seq_len=25):
    result = start_str
    for _ in range(length):
        # Get the last seq_len chars to "warm up" the hidden state
        context = result[-seq_len:]
        x_indices = [char_to_ix[c] for c in context]
        x_seq = one_hot(x_indices, vocab_size)
        
        # Forward pass (starts from 0 hidden state but builds up through the sequence)
        logits = model.forward(x_seq)
        
        # We only take the prediction for the very last character in the window
        probs = head.forward(logits[-1:])
        
        idx = np.random.choice(range(vocab_size), p=probs.ravel())
        result += ix_to_char[idx]
    return result

# 5. INITIALIZATION
hidden_dims = [128, 128]
seq_len = 30  # Increased for better context
lr = 0.01     # Stable learning rate
model = StackedLSTM(vocab_size, hidden_dims, vocab_size, lr=lr)
head = SoftmaxHead()

# 6. TRAINING LOOP
print("Starting training... This might take a few minutes.")
for epoch in range(20_000):
    # Sequential-ish random sampling
    start_idx = np.random.randint(0, len(text) - seq_len - 1)
    chunk = text[start_idx : start_idx + seq_len + 1]
    
    inputs = encode(chunk[:-1])
    targets = encode(chunk[1:])
    
    # Forward -> Loss -> Backward -> Update
    x_seq = one_hot(inputs, vocab_size)
    logits = model.forward(x_seq)
    probs = head.forward(logits)
    
    loss, d_logits = head.loss(probs, targets)
    model.backward(d_logits)
    model.update()
    
    # Adaptive Learning Rate (Annealing)
    if epoch == 8000:
        model.lr = 0.005
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        # Sample with a known "seed" word
        test_output = sample(model, head, "the ", 50, seq_len=seq_len)
        print(f"Sample: {test_output}")
        print("-" * 50)