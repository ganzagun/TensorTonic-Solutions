import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    n_test, d = X_test.shape

    y_test = []

    for X_curr in X_test:
        X_curr = X_curr.reshape(1, d)
        dist = np.sum((X_train - X_curr)**2, axis = 1)
        dist = np.sqrt(dist)
        top_idx = np.argpartition(dist, k-1)[:k]

        values, counts = np.unique(y_train[top_idx], return_counts=True)
        y_cur = values[np.argmax(counts)]
        y_test.append(y_cur)

    return np.array(y_test)

    
