import numpy as np

def gbr_predict(X_train, y_train, X_test, n_estimators=10, max_depth=3, learning_rate=0.1, seed=42):
    """
    Returns: list of predicted values rounded to 4 decimal places
    """

    def mse(y):
        yh = np.mean(y)
        return np.mean((y-yh)**2)

    def best_split(X, y):
        best_feat = None
        best_thres = None 
        best_score = -np.inf
        n, d = X.shape

        parent_score = mse(y)

        for feat in range(d):
            possible = np.unique(X[:, feat])
            for thres in possible:
                left_mask, right_mask = X[:, feat] <= thres, X[:, feat] > thres
                n_left , n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue 
                curr_score = parent_score - (n_left/n)*mse(y[left_mask]) - (n_right/n)*mse(y[right_mask])

                if curr_score > best_score:
                    best_score = curr_score
                    best_feat = feat 
                    best_thres = thres
        return best_feat, best_thres, best_score 


    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) == 1:
            return {'leaf': True, 'label': np.mean(y)}

        cur_feat, cur_thres, cur_score = best_split(X, y)

        if cur_feat is None or cur_score <= 0:
            return {'leaf': True, 'label': np.mean(y)}

        left_mask = X[:, cur_feat] <= cur_thres

        return {
            'leaf': False, 'thres': cur_thres, 'feat': cur_feat,
            'left': build_tree(X[left_mask], y[left_mask], depth+1),
            'right': build_tree(X[~left_mask], y[~left_mask], depth+1)
        }
    def predict(node, x):
        if node['leaf']:
            return node['label']
        if x[node['feat']] <= node['thres']:
            return predict(node['left'], x)
        return predict(node['right'], x)

    trees = []
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    n = X_train.shape[0]

    F0 = np.mean(y_train)
    F = np.full(n, F0)

    for _ in range(n_estimators):
        residual = y_train - F
        tree = build_tree(X_train, residual, 0)
        trees.append(tree)
        for i in range(n):
            F[i] += learning_rate * predict(tree, X_train[i])

    preds = [] 

    for x in X_test:
        cur_pred = F0 
        for tree in trees:
            cur_pred += learning_rate*predict(tree, x)
        preds.append(round(cur_pred, 4))

    return np.array(preds)
