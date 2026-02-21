import numpy as np
from recurrent.gru import GRUS

# ! this code is ai generated

def string_to_onehot(string, char_to_idx, vocab_size):
    # Converts a string into a sequence of one-hot vectors (T, Vocab_Size)
    seq = np.zeros((len(string), vocab_size))
    for i, char in enumerate(string):
        seq[i, char_to_idx[char]] = 1
    return seq

# Setup Data
text = "If you want that GRU to feel as 'smooth' as the RNN did, the Dense Layer with"

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}, Chars: {chars}")

# Hyperparameters
hidden_dim = 32
learning_rate = 0.001
epochs = 15000


#! Since the final ;;ayer os npt part of the rms prop momentum... 
# the two parts of the model that are the recurrent units and the final linear projection are not converging better 
# We add the final projection to the rms prop too

# Initialize Layers
gru = GRUS(input_dim=vocab_size, hidden_dim=hidden_dim, lr=learning_rate)
# Final output layer: turns hidden state (H) into character scores (Vocab)
W_out = np.random.randn(hidden_dim, vocab_size) * 0.01
b_out = np.zeros((1, vocab_size))

# Training loop
for epoch in range(epochs):
    # 1. Forward Pass
    x_input = string_to_onehot(text[:-1], char_to_idx, vocab_size) # "hello worl"
    y_true = string_to_onehot(text[1:], char_to_idx, vocab_size)  # "ello world"
    
    # GRU pass
    h_states = gru.forward(x_input) # (T, H)
    
    # Output pass
    logits = np.dot(h_states, W_out) + b_out # (T, Vocab)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    
    # 2. Calculate Loss (Cross-Entropy)
    loss = -np.sum(y_true * np.log(probs + 1e-8)) / len(text)
    
    # 3. Backward Pass
    d_logits = (probs - y_true) / len(text) # (T, Vocab)
    
    # Gradient for Output weights
    dW_out = np.dot(h_states.T, d_logits)
    db_out = np.sum(d_logits, axis=0, keepdims=True)
    
    # Gradient to pass into GRU
    dh_in = np.dot(d_logits, W_out.T)
    
    # Backprop through GRU
    gru.backward(dh_in)
    
    # Update Output weights (Simple SGD for simplicity here)
    W_out -= learning_rate * dW_out
    b_out -= learning_rate * db_out
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nTraining Complete!")

def generate(start_char, length):
    curr_char = start_char
    result = [curr_char]
    
    for _ in range(length):
        x = string_to_onehot(curr_char, char_to_idx, vocab_size)
        h = gru.forward(x)
        logits = np.dot(h, W_out) + b_out
        next_idx = np.argmax(logits[-1])
        curr_char = idx_to_char[next_idx]
        result.append(curr_char)
    
    return "".join(result)

print("Prediction starting with 'h':", generate("h", 10))