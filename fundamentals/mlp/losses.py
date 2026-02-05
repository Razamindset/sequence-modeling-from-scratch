import numpy as np

def mse(y_true, y_pred):
    """MSE: Mean Squared Error"""
    return np.mean(np.square(y_true - y_pred))

def mse_prime(y_true, y_pred):
    """Derivative of the mean squared Error"""
    return 2 * (y_pred - y_true) / np.size(y_true)
