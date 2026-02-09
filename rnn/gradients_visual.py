import matplotlib.pyplot as plt
import numpy as np
from stacked import StackedRNNModel

# --- Setup a deeper test to see the vanishing effect ---
T = 40
num_layers = 6
vocab_size = 10
hidden_size = 32

model = StackedRNNModel([vocab_size] + [hidden_size]*num_layers, vocab_size)
x_test = np.eye(vocab_size)[np.random.randint(0, vocab_size, T)]
targets_test = np.random.randint(0, vocab_size, T)

# Get the norms
grad_norms = model.backward(model.forward(x_test), targets_test)

# Convert to clean numpy array for plotting
data_to_plot = np.array(grad_norms, dtype=float)

plt.figure(figsize=(12, 6))
plt.imshow(data_to_plot, aspect='auto', cmap='magma', interpolation='nearest')
plt.colorbar(label='Gradient Norm (Signal Strength)')
plt.title('Vanishing Gradient Heatmap: Layer vs Time')
plt.ylabel('Layer (0 = Bottom, 5 = Top)')
plt.xlabel('Time Step (0 = Start of sentence)')
plt.show()

# Layer-wise average plot
plt.figure(figsize=(10, 5))
plt.plot(np.mean(data_to_plot, axis=1), marker='s', color='red')
plt.yscale('log')
plt.title('Signal Decay Across Layers (Log Scale)')
plt.xlabel('Layer Index')
plt.ylabel('Mean Gradient Magnitude')
plt.grid(True, which="both", ls="-")
plt.show()