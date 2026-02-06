import numpy as np

sent = ["The", "cat", "sat", "on", "mat"]
sentence = np.eye(5) # Creates a 5x5 Identity matrix (one-hot vectors)

input_size = 5
hidden_size = 3
output_size=5

Wxh = np.random.randn(input_size, hidden_size) * 0.1
Whh = np.random.randn(hidden_size, hidden_size) * 0.1
Why = np.random.randn(hidden_size, output_size) * 0.1

bh = np.zeros((1, hidden_size)) # Hidden bias
by = np.zeros((1, output_size)) # Output bias

# initial hidden state
h = np.zeros((1, hidden_size))

print("Starting forward pass...")

for t in range(len(sentence)):
    x_t = sentence[t].reshape(1, -1) # Current word vector (1, 5)

    # step 1
    # h_t = tanh(x_t @ Wxh + h_prev @ Whh + bh)
    h = np.tanh(np.dot(x_t, Wxh) + np.dot(h, Whh) + bh)

    # Step 2 output projection
    # y_t = Why @ ht + by
    y_t = np.dot(h, Why) + by

    # Softmax to get probabilities
    probabilities = np.exp(y_t) / np.sum(np.exp(y_t))
    
    predicted_word_idx = np.argmax(probabilities)
    
    print(f"Time {t} | Input: Word {sent[t]} | Hidden State: {h.round(2)} | Predicted Index: {sent[predicted_word_idx]}")

print("\n--- Sequence Complete ---")