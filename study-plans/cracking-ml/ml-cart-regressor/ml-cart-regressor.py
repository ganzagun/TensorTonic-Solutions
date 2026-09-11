import numpy as np


def cart_regress(X_train, y_train, X_test, max_depth=5, min_samples=2):
    def mse(y):
        n = len(y)
        if n == 0 :
            return 0.0 
        yh = np.mean(y)

        return np.mean((y - yh)**2)

    def best_split(X, y):
        n, d = X.shape
        best_gain = - np.inf 
        best_feat = None 
        best_thres = None 
        parent_gini = mse(y)

        for feat in range(d):
            possible_thres = np.unique(X[:, feat])
            for thres in possible_thres:
                left_mask = X[:, feat] <= thres 
                right_mask = X[:, feat] > thres
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue 
                gain = parent_gini - (n_left/n) * mse(y[left_mask]) - (n_right/n) * mse(y[right_mask])
                if gain > best_gain:
                    best_gain = gain 
                    best_feat = feat 
                    best_thres = thres

        return best_feat, best_thres, best_gain

    def build_tree(X, y, depth):
        if depth >= max_depth or len(y) < min_samples:
            return { 'leaf': True, 'label': np.mean(y)}

        cur_feat, cur_thres, cur_gain = best_split(X, y)
        if cur_feat is None or cur_gain <= 0:
            return { 'leaf': True, 'label': np.mean(y)}

        left_mask = X[:, cur_feat] <= cur_thres 
        right_mask = X[:, cur_feat] > cur_thres
        return {
            'leaf': False , 'feat': cur_feat, 'thres': cur_thres,
             'left': build_tree(X[left_mask], y[left_mask], depth + 1),
             'right': build_tree(X[right_mask], y[right_mask], depth+1)
        }

    def predict(node, x_test):
        if node['leaf']:
            return node['label']
        if x_test[node['feat']] <= node['thres']:
            return predict(node['left'], x_test)
        else:
            return predict(node['right'], x_test)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    tree = build_tree(X_train, y_train, 0)

    return [predict(tree, x) for x in X_test]
