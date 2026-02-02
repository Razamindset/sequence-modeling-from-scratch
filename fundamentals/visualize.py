import numpy as np
import matplotlib.pyplot as plt
# (Assuming your SGD class is in a file named SGD.py)
from SGD import SGD 

def loss_fn(w):
    # The function we are trying to minimize
    return w[0]**2 + 10 * w[1]**2

def grad_fn(w):
    # The derivative (gradient) of the function above
    return np.array([2 * w[0], 20 * w[1]])

def run_optimizer(optimizer, start_point, n_iters=20):
    w = start_point.copy()
    trajectory = [w.copy()]
    for _ in range(n_iters):
        grads = grad_fn(w)
        params = {"w": w}
        grad_dict = {"w": grads}
        params = optimizer.update(params, grad_dict)
        w = params["w"]
        trajectory.append(w.copy())
    return np.array(trajectory)

# --- Visualization Setup ---
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
# FIX: Match the Z calculation to the loss_fn
Z = X**2 + 10 * Y**2 

optimizer = SGD(lr=0.08) # Slightly lower LR for smoother visual
start = np.array([3.5, 3.5])
trajectory = run_optimizer(optimizer, start)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the "Bowl"
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Plot the Path
traj_z = np.array([loss_fn(p) for p in trajectory])
ax.plot(trajectory[:, 0], trajectory[:, 1], traj_z, color='red', marker='o', markersize=3, label="SGD Path")

ax.set_xlabel("w1 (Weight 1)")
ax.set_ylabel("w2 (Weight 2)")
ax.set_zlabel("Loss")
ax.legend()
plt.show()