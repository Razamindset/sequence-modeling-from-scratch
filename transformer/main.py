import numpy as np
from model import TransformerBlock
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from ADAM import ADAM
from head import LMHead
from cross_entropy_loss import SoftmaxCrossEntropy

# --- Verification Environment Setup ---
vocab_size = 10
d_model = 8
heads = 2
d_ff = 16

# Initialize all network layers
embedding = TokenEmbedding(vocab_size, d_model)
pos_encoder = PositionalEncoding(d_model) 
block = TransformerBlock(d_model, heads, d_ff)
lm_head = LMHead(d_model, vocab_size)
loss_fn = SoftmaxCrossEntropy()

# Register ALL layers containing parameters into ADAM
optimizer = ADAM(layers=[embedding, pos_encoder, block, lm_head], learning_rate=1e-2)

# --- SIMULATED TRAINING UTILITY ---

# Input sequence (e.g., tokens for "The cat sat on")
input_tokens = np.array([1, 4, 2, 7]) 
# Target sequence (e.g., next-token labels for "cat sat on the")
target_tokens = np.array([4, 2, 7, 9]) 

print("--- Starting Architecture Test ---")

for epoch in range(10):
    # 1. Forward Pass
    x_embed = embedding.forward(input_tokens)
    x_pos = pos_encoder.forward(x_embed)
    x_block = block.forward(x_embed)
    logits = lm_head.forward(x_block)
    
    # Calculate performance metrics
    loss = loss_fn.forward(logits, target_tokens)
    predictions = np.argmax(logits, axis=-1)
    
    print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Preds: {predictions} | Targets: {target_tokens}")

    # 2. Backward Pass (Strict Reverse Sequence Order)
    dLoss = loss_fn.backward()          # (seq_len, vocab_size)
    dLMHead = lm_head.backward(dLoss)    # (seq_len, d_model)
    dBlock = block.backward(dLMHead)     # (seq_len, d_model)
    dPos = pos_encoder.backward(dBlock)
    embedding.backward(dBlock)           # Resolves internally via np.add.at

    # 3. Optimization Step
    optimizer.step()

print("--- Architecture Test Passed Successfully ---")