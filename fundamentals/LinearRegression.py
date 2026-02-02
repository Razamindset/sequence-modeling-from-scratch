import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from SGD import SGD

class LinearRegression:
    def __init__(self, optimizer, n_iters=10000):
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.optimizer = optimizer
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # The sixe of the weights is same as that fo the features
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []
        self.param_history = []

        print("W shape: ", self.w.shape)
        loss = 0

        # Training loop
        for epoch in range(self.n_iters):

            pred = np.dot(X, self.w) + self.b
            # (nsamples​,nfeatures​)⋅(nfeatures​) 
            # (nsamples​,)
            # it returns a vector of length n_samples.
            # Each entry = prediction for one sample.

            # now we calculate gradients

            error = pred - y

            # dL/dw
            dw = 1/n_samples * np.dot(X.T, error)
            db = 1/n_samples * np.sum(error)

            # self.w -= self.lr * dw
            # self.b -= self.lr * db

            #* Use and optimizer from our previous implementation
            grads = {'w': dw, 'b': db}
            params = {'w': self.w, 'b': self.b}

            params = self.optimizer.update(params, gradients=grads)

            self.w = params["w"]
            self.b = params["b"]

            self.param_history.append((self.w.copy(), self.b))

            # compute loss
            loss = (1/n_samples) * np.sum(error ** 2)
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_iters}, Loss: {loss:.6f}")

    def predict(self, x):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred

# Prepare the data
X, y = datasets.make_regression(n_samples=1000, n_features=2, noise=2, random_state=47)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LinearRegression(SGD(lr=0.01), n_iters=1000)

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
plt.show()