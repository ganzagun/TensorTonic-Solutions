import numpy as np

def adaboost_classify(X_train, y_train, X_test, n_estimators=10, seed=42):
    """
    Returns: list of predicted labels in {-1, +1} for each test point
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    n, d = X_train.shape 

    def fit_stump(X, y, w):
        best_err = np.inf
        best_feat = None 
        best_thres = None 
        best_polarity = None 

        for feat in range(d):
            for thres in np.unique(X[:, feat]):
                for polarity in [ 1, -1]:
                    preds = np.ones(n)
                    if polarity == 1:
                        preds[X[:, feat] <= thres] = -1
                    else :
                        preds[X[:, feat] > thres] = - 1

                    error = np.sum(w[preds != y])

                    if error < best_err:
                        best_err = error 
                        best_feat = feat 
                        best_thres = thres 
                        best_polarity = polarity
        return best_feat, best_thres, best_polarity, best_err

    def stump_pred(X, feat, thres, polarity):
        preds = np.ones(X.shape[0])
        if polarity == 1:
            preds[X[:, feat] <= thres] = -1;
        else:
            preds[X[:, feat] > thres] = -1
        return preds 

    W = np.ones(n)/ n
    stumps = []
    alphas = []

    for _ in range(n_estimators):
        cur_feat, cur_thres, cur_polarity, cur_err = fit_stump(X_train, y_train, W)
        cur_err = np.clip(cur_err, 1e-10, 1 - 1e-10)
        alpha = (1/2) * np.log((1-cur_err)/cur_err)
        preds = stump_pred(X_train,  cur_feat, cur_thres, cur_polarity)
        W = W* np.exp(-alpha * y_train * preds)
        W /= np.sum(W)
        stumps.append((cur_feat, cur_thres, cur_polarity))
        alphas.append(alpha)

    y_preds = []

    for x in X_test:
        cur_score = 0.0
        for (feat, thres, polarity), alpha in zip(stumps, alphas):
            if polarity == 1:
                preds = -1 if x[feat] <= thres else 1
            else:
                preds = -1 if x[feat] > thres else 1
            cur_score += alpha*preds 

        y_preds.append(1 if cur_score >= 0 else -1)

    return np.array(y_preds)
        