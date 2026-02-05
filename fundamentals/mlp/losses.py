import numpy as np

def mse(y_true, y_pred):
    """MSE: Mean Squared Error"""
    return np.mean(np.square(y_true - y_pred))

def mse_prime(y_true, y_pred):
    """Derivative of the mean squared Error"""
    return 2 * (y_pred - y_true) / np.size(y_true)

def cross_entropy(y_true, y_pred):
    # Clip values to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1. - 1e-15)
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    # This assumes the layer before this was Softmax
    return y_pred - y_true