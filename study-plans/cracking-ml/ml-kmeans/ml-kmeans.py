import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    X = np.array(X, dtype=float)
    rng = np.random.RandomState(seed)
    n, d = X.shape
    initial_idx = rng.choice(n, size = k, replace = False)

    means = X[initial_idx]

    for _ in range(max_iters):
        old_means = means.copy()
        dist = np.sum((X[:, None, :] - means[None, :, :])**2, axis = 2)
        assigned_points = np.argmin(dist, axis = 1)
        for i in range(k):
            points = X[assigned_points == i]

            if len(points) > 0:
                means[i] = np.mean(points, axis=0)

        if np.allclose(means, old_means):
            break


    return (
        assigned_points.tolist(),
        np.round(means, 4).tolist()
    )
