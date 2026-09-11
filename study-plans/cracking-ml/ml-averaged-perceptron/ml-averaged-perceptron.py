import numpy as np

def averaged_perceptron(X_train, y_train, X_test, n_epochs=10):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train) 
    X_test = np.asarray(X_test)
    n, d = X_train.shape

    w_all = []
    b_all = []

    W = np.zeros(d)
    b = 0.0

    for epoch in range(n_epochs):
        for x, y in zip(X_train, y_train):
            if y*(x@W + b) <= 0:
                W = W + y*x
                b = b + y
            w_all.append(W)
            b_all.append(b)

    w_all = np.array(w_all)
    b_all = np.array(b_all)

    w_all = np.mean(w_all, axis = 0)
    b_all = np.mean(b_all)

    y_test = X_test@w_all + b_all
    return  np.where(y_test > 0, 1, -1)
    
