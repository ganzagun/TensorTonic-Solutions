import numpy as np

def auc_roc(y_true, y_scores):
    """
    Returns: tuple of (fpr_list, tpr_list, auc_value)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    desc_idx = np.argsort(-y_scores)
    y_true = y_true[desc_idx]
    y_scores = y_scores[desc_idx]

    thresholds = np.unique(y_scores)[::-1]

    tprs = [0.0]
    fprs = [0.0]
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        predicted_pos = y_scores >= thresh
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        tpr = tp / P if P > 0 else 0
        fpr = fp / N if N > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2.0

    return ([round(float(v), 4) for v in fprs],
            [round(float(v), 4) for v in tprs],
            round(float(auc), 4))
