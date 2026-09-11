import numpy as np

#log-sum-exp trick
def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    X = np.array(X)
    y = np.array(y)
    y_onehot = np.eye(n_classes)[y]
    n,d = X.shape

    W = np.zeros((d, n_classes))
    b = np.zeros(n_classes)

    for ep in range(n_iters):
        yh = softmax(X@W + b)

        dW = (1/n) * X.T@(yh - y_onehot)
        db = (1/n) * np.sum(yh - y_onehot, axis=0)

        W -= lr*dW 
        b -=lr*db 

    return W, b
