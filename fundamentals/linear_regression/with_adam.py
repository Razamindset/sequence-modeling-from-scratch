import numpy as np

#! The following code uses Mini Batch Gradient decent with ADAM for leaning rate control
# Weight decay adds a "cost" to the size of the weights. The model now has two goals:
#   1. Minimize the prediction error (MSE).
#   2. Keep the weights as small as possible.
#   total Loss = {MSE} + sum(w**2)* lambda/(2) 

tol = 1e-6
patience = 14

 
class LinearRegressionAdam:
    def __init__(self, lr=0.001, n_iters=10000, batch_size=32, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.w, self.b = None, None
        self.batch_size = batch_size
        self.losses = []
    
    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape

        # The sie of the weights is same as that for the features
        self.w = np.zeros(n_features)
        self.b = 0

        # Early Stopping variables
        self.best_loss = float('inf')
        wait = 0

        # Adam stuff
        m_w, v_w = np.zeros(n_features), np.zeros(n_features)
        m_b, v_b = 0, 0
        t = 0 # this is usesin early bias correction

        # Training loop
        for epoch in range(self.n_iters):

            # Shuffle the data 
            indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[indices], y[indices]

            epoch_loss = 0

            for i in range(0, n_samples, self.batch_size):
                t += 1
                # Grab a chunk of data
                xi = X_shuffled[i : i + self.batch_size]
                yi = y_shuffled[i : i + self.batch_size]

                pred = np.dot(xi, self.w) + self.b
                error = pred - yi

                dw = (1 / len(xi)) * np.dot(xi.T, error)
                db = (1 / len(xi)) * np.sum(error)

                # simple weight decay
                # The derivative of (lambda/2 * w^2) is (lambda * w)
                # see notes for details
                dw += self.weight_decay * self.w


                # first moment , momentum
                m_w = self.beta1 * m_w + (1-self.beta1) * dw
                m_b = self.beta1 * m_b + (1-self.beta1) * db

                # second moment , velocity
                v_w = self.beta2 * v_w + (1-self.beta2) * dw **2
                v_b = self.beta2 * v_b + (1-self.beta2) * db **2

                # Bias correction durin early steps...
                m_w_corr = m_w / (1 - self.beta1**t)
                m_b_corr = m_b / (1 - self.beta1**t)

                v_w_corr = v_w / (1 - self.beta2**t)
                v_b_corr = v_b / (1 - self.beta2**t)

                # Update rule
                self.w -= self.lr * m_w_corr / (np.sqrt(v_w_corr) + self.epsilon)
                self.b -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)

                epoch_loss += np.sum(error**2)

           
            # compute loss
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
                print(f"Epoch {epoch+1}/{self.n_iters}, Loss: {self.losses[-1]:.6f}")

    def predict(self, x):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred
