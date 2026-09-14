import numpy as np

def log_loss(y_true, y_pred):
    """
    Returns: float
    """
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    n = y_true.shape[0]
    loss  = - (1/n) * np.sum(
        y_true* np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
    )

    return loss
