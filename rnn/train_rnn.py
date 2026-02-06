import numpy as np
from rnn import VanillaRNN as RNN

# 1. Prepare Data
phrases = [
    "some cat sat on mat",
    "the dog ran in park",
    "a bird flew over tree"
]

# Create Vocabulary
words = sorted(list(set(" ".join(phrases).split())))
word_to_idx = {w: i for i, w in enumerate(words)}
idx_to_word = {i: w for i, w in enumerate(words)}
vocab_size = len(words)

print(f"Vocab Size: {vocab_size} | Words: {words}")

# 2. Initialize Model
# We'll use a larger hidden_size to help it distinguish between 'the cat' and 'the dog'
hidden_dim = 16
rnn = RNN(input_dim=vocab_size, hidden_dim=hidden_dim, output_dim=vocab_size, lr=0.1)

# 3. Training Loop
epochs = 300
for epoch in range(epochs + 1):
    total_loss = 0
    
    # Shuffle phrases so the model doesn't just memorize the order of phrases
    np.random.shuffle(phrases)
    
    for phrase in phrases:
        tokens = phrase.split()
        # Convert tokens to one-hot vectors
        x_seq = []
        targets = []
        for i in range(len(tokens) - 1):
            one_hot = np.zeros(vocab_size)
            one_hot[word_to_idx[tokens[i]]] = 1
            x_seq.append(one_hot)
            targets.append(word_to_idx[tokens[i+1]]) # Target is the NEXT word
            
        x_seq = np.array(x_seq)
        
        # Forward, Backward, Update
        h_states, y_probs = rnn.forward(x_seq)
        
        # Calculate Loss
        loss = sum(-np.log(y_probs[t][0, targets[t]] + 1e-8) for t in range(len(targets)))
        total_loss += loss
        
        grads = rnn.backward(x_seq, h_states, y_probs, targets)
        rnn.update_params(grads)
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Total Loss: {total_loss:.4f}")

# 4. Creative Generation Function
def generate(model, start_word, length=5, temperature=1.0):
    current_word = start_word
    sentence = [current_word]
    h = np.zeros((1, model.hidden_dim))
    
    for _ in range(length):
        if current_word not in word_to_idx: break
        
        # Prepare Input
        x = np.zeros((1, vocab_size))
        x[0, word_to_idx[current_word]] = 1
        
        # Single Step Forward (Manual)
        h = np.tanh(np.dot(x, rnn.Wxh) + np.dot(h, rnn.Whh) + rnn.bh)
        y = np.dot(h, rnn.Why) + rnn.by
        
        # Apply Temperature (Higher = more random, Lower = more predictable)
        # p = softmax(y / temp)
        exp_y = np.exp(y / temperature)
        probs = exp_y / np.sum(exp_y)
        
        # Sample from the distribution (instead of just taking argmax)
        next_idx = np.random.choice(range(vocab_size), p=probs.ravel())
        
        current_word = idx_to_word[next_idx]
        sentence.append(current_word)
        
    return " ".join(sentence)

print("\n--- Testing Generation ---")
print("Seed 'the':", generate(rnn, "the", length=4, temperature=0.7))
print("Seed 'a':  ", generate(rnn, "a", length=4, temperature=0.7))
print("Seed 'some':  ", generate(rnn, "some", length=4, temperature=0.7))