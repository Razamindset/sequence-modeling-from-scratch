import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

#! The following code uses Mini Batch Gradient decent 

class LinearRegressionMiniBatch:
    def __init__(self, lr=0.001, n_iters=10000, batch_size=32):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.batch_size = batch_size
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # The sie of the weights is same as that for the features
        self.w = np.zeros(n_features)
        self.b = 0

        # Training loop
        for epoch in range(self.n_iters):

            # Shuffle the data 
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, self.batch_size):
                # Grab a chunk of data
                xi = X_shuffled[i : i + self.batch_size]
                yi = y_shuffled[i : i + self.batch_size]

                pred = np.dot(xi, self.w) + self.b
                error = pred - yi

                dw = (1 / len(xi)) * np.dot(xi.T, error)
                db = (1 / len(xi)) * np.sum(error)

                # 5. Update
                self.w -= self.lr * dw
                self.b -= self.lr * db

                epoch_loss += np.sum(error**2)

           
            # compute loss
            self.losses.append(epoch_loss / n_samples)

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_iters}, Loss: {self.losses[-1]:.6f}")

    def predict(self, x):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred

# Prepare the data
X, y = datasets.make_regression(n_samples=10_000, n_features=4, noise=2, random_state=47)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LinearRegressionMiniBatch(n_iters=1000, lr=0.01)
model.fit(X_train, y_train)

plt.figure()
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (MSE)")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()

# Make predictions
predictions = model.predict(X_test)

plt.figure()
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("True vs Predicted")
plt.grid(True)
plt.show()
