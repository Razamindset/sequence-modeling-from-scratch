import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    sig =  sigmoid(z)
    return sig * (1 - sig)

def binary_cross_entropy(y_true, y_pred):
    """
    y_true: shape (m,)
    y_pred: shape (m,)  -- predicted probabilities (after sigmoid)
    """

    epsilon = 1e-15  # prevents log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )

    return loss

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # The sixe of the weights is same as that fo the features
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        print("W shape: ", self.w.shape)
        loss = 0

        # Training loop
        for epoch in range(self.n_iters):

            pred = self.predict(X)
            # (nsamples​,nfeatures​)⋅(nfeatures​) 
            # (nsamples​,)
            # it returns a vector of length n_samples.
            # Each entry = prediction for one sample.

            # now we calculate gradients

            error = pred - y

            # dL/dw
            dw = 1/n_samples * np.dot(X.T, error)
            db = 1/n_samples * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # compute loss
            loss = binary_cross_entropy(y, pred)
            self.losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{self.n_iters}, Loss: {loss:.6f}")

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        y_pred = sigmoid(z)
        return y_pred
    
# Prepare the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

#* Introducing a scaler has increased the accuracy from 89 to 97% --- good stuff
# without scaling the model:
# Zig-zags
# Takes unstable steps
# Struggles to converge
# Needs very small learning rate
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

model = LogisticRegression()
model.fit(X_train, y_train)

plt.figure()
plt.plot(model.losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss (Binary Cross Entropy)")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()

# Make predictions
predictions = model.predict(X_test)

# Convert probabilities to class labels
pred_labels = (predictions >= 0.5).astype(int)

accuracy = np.mean(pred_labels == y_test)
print("Accuracy:", accuracy)


plt.figure()
plt.hist(predictions[y_test == 0], bins=20, alpha=0.6, label="Class 0")
plt.hist(predictions[y_test == 1], bins=20, alpha=0.6, label="Class 1")

plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Prediction Distribution by True Class")
plt.legend()
plt.grid(True)
plt.show()
