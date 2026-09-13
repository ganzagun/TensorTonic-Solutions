import numpy as np

def kmeans_plusplus(X, k, seed=42):
    X = np.asarray(X, dtype = float)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    first_idx = rng.randint(0, n)
    means = []
    means.append(X[first_idx])

    for _ in range(k-1):
        np_means = np.array(means)
        dist = np.mean(
            (X[:, None, :] - np_means[None, :, :])**2, axis = 2
        )
        closest_dist = np.min(dist, axis=1)
        prob = closest_dist/ np.sum(closest_dist)
        next_idx = rng.choice(n, p = prob)
        means.append(X[next_idx])

    return [np.round(mean, 4) for mean in means]
        
