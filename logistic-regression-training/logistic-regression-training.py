import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    D, N = X.shape[1], X.shape[0]

    W = np.zeros(D)
    b = np.zeros(1)

    for epoch in range(steps):
        z = X @ W + b
        y_hat = _sigmoid(z)
        error = y_hat - y

        dw = (X.T @ error) / N
        db = np.sum(error) / N

        W -= lr * dw
        b -= lr * db


    return W, b