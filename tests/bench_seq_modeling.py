from recurrent.sequential import SequentialModel
from recurrent.rnn import RNNLayer
from recurrent.dense import Dense as OutputLayer
from recurrent.gru import GRU
from recurrent.lstm import LSTMLayer

import matplotlib.pyplot as plt
import numpy as np
import time

# Expanded Dataset for Research
text = """
The artificial intelligence revolution is transforming how we write code. 
Deep learning models like RNNs, GRUs, and LSTMs allow machines to understand sequences. 
While a standard RNN is fast, it struggles with long-term memory. 
The GRU and LSTM were designed to solve this by using gates to control information flow. 
Next, we will explore Transformers and Attention mechanisms.
""".strip().lower()

# Re-run your preparation logic
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def prepare_data(string):
    # Convert string to indices for target and one-hot for input
    indices = [char_to_idx[ch] for ch in string]
    one_hots = np.eye(vocab_size)[indices]
    return one_hots, indices

x_data, y_targets = prepare_data(text)
x_train = x_data[:-1]
y_train = y_targets[1:]

def train_with_logging(model, x_full, y_full, window_size=32, epochs=300):
    losses = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        num_chunks = 0

        for i in range(0, len(x_full) - window_size, window_size):
            x_chunk = x_full[i:i+window_size]
            y_chunk = y_full[i:i+window_size]

            loss = model.train_step(x_chunk, y_chunk)
            epoch_loss += loss
            num_chunks += 1

        avg_loss = epoch_loss / num_chunks
        losses.append(avg_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    return losses, total_time

            
# --- Setup Iso-parameter Architecture ---
# Target: ~17k-18k parameters total
lstm_h = 64
gru_h = 74
rnn_h = 128 

lr = 0.001

# Initialize 1: Wide RNN (Matches LSTM capacity)
rnn_layers = [RNNLayer(vocab_size, rnn_h)]
rnn_model = SequentialModel(rnn_layers, OutputLayer(rnn_h, vocab_size), lr=lr)

# Initialize 2: Medium GRU
gru_layers = [GRU(vocab_size, gru_h)]
gru_model = SequentialModel(gru_layers, OutputLayer(gru_h, vocab_size), lr=lr)

# Initialize 3: Standard LSTM
lstm_layers = [LSTMLayer(vocab_size, lstm_h)]
lstm_model = SequentialModel(lstm_layers, OutputLayer(lstm_h, vocab_size), lr=lr)

epochs = 250

print("\nTraining RNN...")
rnn_losses, rnn_time = train_with_logging(rnn_model, x_train, y_train, epochs=epochs)

print("\nTraining GRU...")
gru_losses, gru_time = train_with_logging(gru_model, x_train, y_train, epochs=epochs)

print("\nTraining LSTM...")
lstm_losses, lstm_time = train_with_logging(lstm_model, x_train, y_train, epochs=epochs)

def count_parameters(model, model_name):
    total = 0
    # Add hidden layer params
    for layer in model.layers:
        for name, (p, grad) in layer.get_params().items():
            total += p.size
    # Add output layer params
    for name, (p, grad) in model.output_layer.get_params().items():
        total += p.size
    print(f"{model_name} Total Parameters: {total}")

rnn_params_count = count_parameters(rnn_model, "Wide-RNN")
gru_params_count = count_parameters(gru_model, "Mid-GRU")
lstm_params_count = count_parameters(lstm_model, "LSTM")

print("\n==== FINAL COMPARISON ====")
print(f"RNN Time: {rnn_time:.2f} sec")
print(f"GRU Time: {gru_time:.2f} sec")
print(f"LSTM Time: {lstm_time:.2f} sec")

print(f"Final RNN Loss: {rnn_losses[-1]:.4f}")
print(f"Final GRU Loss: {gru_losses[-1]:.4f}")
print(f"Final LSTM Loss: {lstm_losses[-1]:.4f}")

plt.figure()
plt.plot(rnn_losses)
plt.plot(gru_losses)
plt.plot(lstm_losses)
plt.title("RNN vs GRU vs LSTM Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["RNN", "GRU", "LSTM"])
plt.show()