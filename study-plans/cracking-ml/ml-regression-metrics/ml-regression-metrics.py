import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Returns: dict with keys "mse", "mae", "r2" rounded to 4 decimal places
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }
