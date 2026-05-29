import matplotlib.pyplot as plt
import numpy as np

# Assuming your modules are in the same folder or package
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from model import TransformerBlock
from head import LMHead
from cross_entropy_loss import SoftmaxCrossEntropy
from ADAM import ADAM

# =====================================================================
# 1. DATA PROCESSING & TOKENIZATION
# =====================================================================

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_id[c] for c in s]
def decode(l): return "".join([id_to_char[i] for i in l])

# Encode the entire dataset into a single long NumPy array
dataset_ids = np.array(encode(text), dtype=np.int32)

print(f"Dataset Loaded. Total characters: {len(text)}")
print(f"Vocabulary Size: {vocab_size} unique characters.")

# =====================================================================
# 2. HYPERPARAMETERS & INITIALIZATION
# =====================================================================
d_model = 128        # Dimension of token vectors
heads = 4           # Number of attention heads (d_k = 64 // 4 = 16)
d_ff = 2*d_model          # Hidden layer size inside FeedForward
seq_len = 24        # Context window size (how many characters the model looks at)
learning_rate = 3e-4
epochs = 200
steps_per_epoch = max(1, (len(dataset_ids) - seq_len - 1) // seq_len)
rng = np.random.default_rng(42)

# Instantiate all layers
embedding  = TokenEmbedding(vocab_size, d_model)
pos_encode = PositionalEncoding(d_model, max_len=100)
block      = TransformerBlock(d_model, heads, d_ff)
lm_head    = LMHead(d_model, vocab_size)
loss_fn    = SoftmaxCrossEntropy()

# Register layers containing parameters to ADAM
optimizer = ADAM(layers=[embedding, block, lm_head], learning_rate=learning_rate)

# =====================================================================
# 3. CORE TRAINING LOOP
# =====================================================================
# Initialize loss tracker before the loop
loss_history = []

print("\n--- Starting Training on Text Data ---")
for epoch in range(epochs):
    epoch_loss = 0.0
    steps = 0
    
    for _ in range(steps_per_epoch):
        i = rng.integers(0, len(dataset_ids) - seq_len - 1)
        input_chunk  = dataset_ids[i : i + seq_len]
        target_chunk = dataset_ids[i + 1 : i + seq_len + 1]
        
        x = embedding.forward(input_chunk)
        x = pos_encode.forward(x)
        x = block.forward(x)
        logits = lm_head.forward(x)
        
        loss = loss_fn.forward(logits, target_chunk)
        epoch_loss += loss
        steps += 1
        
        dLoss   = loss_fn.backward()
        dHead   = lm_head.backward(dLoss)
        dBlock  = block.backward(dHead)
        dPos    = pos_encode.backward(dBlock)
        embedding.backward(dPos)
        
        optimizer.step()
        
    avg_loss = epoch_loss / steps
    loss_history.append(avg_loss) # <-- Save loss for graphing
    print(f"Epoch {epoch+1:02d}/{epochs} | Average Loss: {avg_loss:.4f}")

def generate_text(seed_string, max_new_tokens=200, temperature=0.7, sample=False):
    """
    Generates text character-by-character starting from a seed string.
    Temperature scales the logits: 
      - Low (< 0.5): Conservative, highly repetitive.
      - High (> 1.0): Creative, chaotic, higher chance of typos.
    """
    # Convert seed text into integer IDs
    generated_ids = encode(seed_string)
    
    for _ in range(max_new_tokens):
        # Slice context window to match the exact trained seq_len
        context_ids = generated_ids[-seq_len:]
        input_chunk = np.array(context_ids, dtype=np.int32)
        
        # --- Forward Pass Only ---
        x = embedding.forward(input_chunk)
        x = pos_encode.forward(x)
        x = block.forward(x)
        logits = lm_head.forward(x)
        
        # Isolate the logits for the absolute final token in the sequence
        last_token_logits = logits[-1, :]
        
        if not sample:
            next_token_id = np.argmax(last_token_logits)
            generated_ids.append(next_token_id)
            continue

        # Apply temperature scaling and stable softmax
        scaled_logits = last_token_logits / temperature
        shifted_logits = scaled_logits - np.max(scaled_logits)
        probs = np.exp(shifted_logits) / np.sum(np.exp(shifted_logits))
        
        next_token_id = np.random.choice(len(probs), p=probs)
        
        # Append token to running context
        generated_ids.append(next_token_id)
        
    return decode(generated_ids)

# =====================================================================
# 4. PLOT TRAINING RESULTS & INFERENCE
# =====================================================================

# Generate Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, color='blue', marker='o', linestyle='-')
plt.title("Transformer Training Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Cross Entropy Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("loss_curve.png") # Saves to project directory
plt.show()

# Run Generation Test
print("\n--- Running Inference Test ---")
prompt = "Remember March, the ides of March remember:\n"
generated_output = generate_text(seed_string=prompt, max_new_tokens=150, temperature=0.6)
print(generated_output)
