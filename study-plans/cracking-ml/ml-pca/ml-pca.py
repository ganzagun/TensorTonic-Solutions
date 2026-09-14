import numpy as np

def pca(X, n_components=2):
    """
    Returns: tuple of (transformed_data, explained_variance_ratios)
    """
    X = np.asarray(X, dtype = float)
    n, d = X.shape 

    Xc = X - np.mean(X, axis = 0)

    cov = (Xc.T@Xc)/ (n - 1)
    eigenvalue, eigenvec = np.linalg.eigh(cov)
    sort_idx = np.flip(np.argsort(eigenvalue))
    eigenvalue = eigenvalue[sort_idx]
    eigenvec = eigenvec[:, sort_idx]

    components = eigenvec[:, :n_components]
    Xt = Xc @ components

    explained = eigenvalue[:n_components]/eigenvalue.sum()

    return (
        np.round(Xt, 4).tolist(),
        np.round(explained, 4).tolist(),
    )
    
