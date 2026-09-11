import numpy as np

def svm_sgd(X_train, y_train, X_test, lr=0.01, lam=0.01, n_epochs=100):
    """
    Train an SVM using SGD on hinge loss with L2 regularization.

    Parameters:
    - X_train: Training feature matrix (n samples, d features)
    - y_train: Training labels (-1 or +1)
    - X_test: Test feature matrix
    - lr: Learning rate
    - lam: L2 regularization strength
    - n_epochs: Number of training epochs

    Returns: list of predicted labels (-1 or +1) for each test point
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_test = np.asarray(X_test, dtype=float)
    n, d = X_train.shape
    W = np.zeros(d)
    b = 0.0

    for ep in range(n_epochs):
        for x, y in zip(X_train, y_train):
            margin = y*(x@W + b)
            if margin < 1:
                W = W - lr * ( lam *W - y*x)
                b = b + lr * y 
            else:
                W = W - lr*lam*W

    pred = X_test@W + b 
    return np.where(pred > 0 , 1, -1)