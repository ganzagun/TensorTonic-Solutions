def lasso_regression(X, y, lr, epochs, alpha):
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = float)

    n,d = X.shape
    W = np.zeros(d)
    b = 0.0 

    for ep in range(epochs):
        yh = X@W + b

        dW = (2/n)* X.T@(yh-y) + alpha*np.sign(W)
        db = (2/n)*np.sum(yh-y)

        W -= lr*dW
        b -= lr*db

    return W, b
    