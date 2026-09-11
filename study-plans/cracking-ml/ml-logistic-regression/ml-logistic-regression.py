import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)

    n, d = X.shape

    W = np.zeros(d)
    b = 0.0

    for epoch in range(n_iters):
        yh = sigmoid(X@W + b)
        dW = (1/n) * X.T@(yh-y)
        db = (1/n) * np.sum(yh-y)
        W -= lr*dW
        b -= lr*db 
    return W, b
