def ridge_regression(X, y, lr, epochs, alpha):
    X = np.array(X)
    y = np.array(y)
    n, d = X.shape 

    W = np.zeros(d)
    b = 0.0 

    for epoch in range(epochs):
        yh = X@W + b 
        dW = (2/n)* X.T@(yh - y) + 2*alpha*W
        db = (2/n) * np.sum(yh - y)

        W -= lr*dW 
        b -= lr*db
    return W, b