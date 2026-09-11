import numpy as np

def permutation_importance(X, y, predict_fn, n_repeats=5, seed=42):
    X = np.asarray(X)
    y = np.asarray(y)
    d = X.shape[1]
    rn = np.random.RandomState(seed)

    baseline = np.mean(predict_fn(X) == y)
    imp = np.zeros(d)

    for feat in range(d):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, feat] = rn.permutation(X_perm[:, feat])
            score = np.mean(predict_fn(X_perm) == y)
            scores.append(baseline - score)
        imp[feat] = np.mean(scores)

    return [round(float(x), 4) for x in imp]
