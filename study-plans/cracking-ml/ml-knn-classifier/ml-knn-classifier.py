import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    dist = np.sum(
        (X_train[None, :, :] - X_test[:, None, :])**2,
        axis = 2
    )

    top_idx = np.argpartition(dist, k-1, axis = 1)[:, :k]
    y_test = []

    for idx in top_idx:
        values, count = np.unique(y_train[idx], return_counts = True)
        y_test.append(values[np.argmax(count)])

    return np.asarray(y_test)
