import numpy as np

def precision_recall_ap(y_true, y_scores):
    """
    Returns: tuple of (recall_list, precision_list, ap_value)
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    desc_idx = np.argsort(-y_scores)
    y_true = y_true[desc_idx]
    y_scores = y_scores[desc_idx]

    thresholds = np.unique(y_scores)[::-1]

    P = np.sum(y_true == 1)
    recalls = [0.0]
    precisions = [1.0]

    for thresh in thresholds:
        predicted_pos = y_scores >= thresh
        tp = np.sum(predicted_pos & (y_true == 1))
        fp = np.sum(predicted_pos & (y_true == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / P if P > 0 else 0
        recalls.append(recall)
        precisions.append(precision)

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i-1]) * precisions[i]

    return ([round(float(v), 4) for v in recalls],
            [round(float(v), 4) for v in precisions],
            round(float(ap), 4))
