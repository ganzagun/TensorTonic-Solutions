import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)

    n_test, d = X_test.shape

    y_test = []

     # Calculate squared Euclidean distance
    # between every test point and every training point
    dist = np.sum(
        (X_test[:, None, :] - X_train[None, :, :]) ** 2,
        axis=2
    )

    top_idx = np.argpartition(dist, k - 1, axis=1)[:, :k]

    y_test = []

    # Majority vote for each test point
    for idx in top_idx:
        values, counts = np.unique(
            y_train[idx],
            return_counts=True
        )
        y_test.append(values[np.argmax(counts)])

    return np.array(y_test)

    
