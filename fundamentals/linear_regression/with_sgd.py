import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import datasets
# from sklearn.model_selection import train_test_split

#! The following code uses socastic gradient decent

tol = 1e-6
patience = 16

class LinearRegressionSGD:
    def __init__(self, lr=0.001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []
        self.best_loss = None
    
    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape

        # The sixe of the weights is same as that fo the features
        self.w = np.zeros(n_features)
        self.b = 0

        # Early Stopping variables
        self.best_loss = float('inf')
        wait = 0

        # Training loop
        for epoch in range(self.n_iters):

            # For each iter we go over each sample one by one and shuffle the data with each iter
            indices = np.random.permutation(n_samples)

            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_loss = 0

            # Loop over each sample

            for i in range(n_samples):
                xi, yi = X_shuffled[i:i+1], y_shuffled[i:i+1]

                # prediction and gradient for each sample
                # pred = np.dot(xi, self.w) - self.b
                error = (np.dot(xi, self.w) - yi)
                dw = xi.T.dot(error).flatten()
                db = np.sum(error)

                self.w -= self.lr * dw
                self.b -= self.lr * db

                epoch_loss += error[0]**2

            # Compute average MSE for this epoch
            current_loss = epoch_loss / n_samples
            self.losses.append(current_loss)

            if current_loss < self.best_loss - tol:
                self.best_loss = current_loss
                wait = 0  # Reset the wait counter
            else:
                wait += 1 # No significant improvement
        
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}. Loss has not improved for {patience} epochs.")
                break

            if epoch % 100 == 0 and verbose:
                print(f"Epoch {epoch}/{self.n_iters}, Loss: {self.losses[-1]:.6f}")

    def predict(self, x):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred

# Prepare the data
# X, y = datasets.make_regression(n_samples=1000, n_features=4, noise=2, random_state=47)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# model = LinearRegressionSGD(n_iters=1000)
# model.fit(X_train, y_train)

# plt.figure()
# plt.plot(model.losses)
# plt.xlabel("Epoch")
# plt.ylabel("Training Loss (MSE)")
# plt.title("Training Loss Over Time")
# plt.grid(True)
# plt.show()

# # Make predictions
# predictions = model.predict(X_test)

# plt.figure()
# plt.scatter(y_test, predictions, alpha=0.6)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("True vs Predicted")
# plt.grid(True)
# plt.show()
