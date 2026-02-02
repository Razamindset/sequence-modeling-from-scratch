import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from linear_regression.full_batch import LinearRegression as BatchLR
from linear_regression.with_sgd import LinearRegressionSGD as SGDLR
from linear_regression.mini_batch import LinearRegressionMiniBatch as MiniBatchLR
from linear_regression.with_adam import LinearRegressionAdam as MiniBatchADAM


def run_benchmark(model, X, y, name):
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"-------------------------------------------")
    print(f"📊 SUMMARY for {name}:")
    print(f"   Best MSE Achieved: {model.best_loss:.6f}")
    print(f"   Total Time:        {duration:.4f}s")
    print(f"   Stop Condition:    Epoch {len(model.losses)}")
    print(f"-------------------------------------------\n")
    return model.losses, duration

# 1. Prepare 100K samples
X, y = datasets.make_regression(n_samples=100000, n_features=10, noise=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Initialize Models (adjust n_iters so they all have a fair chance)
models = {
    "Full Batch": BatchLR(lr=0.01, n_iters=10000),
    "True SGD": SGDLR(lr=0.0001, n_iters=10), # 1 epoch = 100k updates
    "Mini-Batch (64)": MiniBatchLR(lr=0.001, n_iters=100, batch_size=64),
    "Adam mini-Batch (64)": MiniBatchADAM(lr=0.001, n_iters=100, batch_size=64)
}

results = {}
for name, model in models.items():
    losses, duration = run_benchmark(model, X_train, y_train, name)
    results[name] = losses

# 3. Visualization
plt.figure(figsize=(10, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)

plt.xlabel("Epochs / Iterations")
plt.ylabel("MSE Loss")
plt.title("Convergence Comparison: Batch vs SGD vs Mini-Batch")
plt.yscale('log') # Use log scale if losses vary wildly
plt.legend()
plt.grid(True)
plt.show()