import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X, y = np.array(X), np.array(y)
    n, d = X.shape
    W = np.zeros(d)
    b = 0.0
    for epoch in range(epochs):
        yh = X@W + b;
        dW = (2/n) * X.T@(yh - y)
        db = (2/n) * np.sum(yh-y)

        W = W - lr*dW
        b = b - lr*db

    return W, b
    
