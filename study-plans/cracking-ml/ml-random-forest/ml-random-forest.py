import numpy as np

def random_forest_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, max_features='sqrt', seed=42):
    rn = np.random.RandomState(seed)
    
    def gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        impurity = 1
        for c in np.unique(y):
            p = np.sum(y == c)/n 
            impurity -= p*p 

        return impurity

    def best_split(X, y):
        n, d = X.shape 
        best_feat = None 
        best_gain = - np.inf 
        best_thres = None 
        parent_gini = gini(y)

        m = d
        if max_features == 'sqrt':
            m = max(1, int(np.sqrt(d)))
        if max_features == 'log2':
            m = max(1, int(np.log2(d)))

        features = rn.choice(d, size=m, replace=False)

        for feat in features:
            possible_thres = np.unique(X[:, feat])
            for thres in possible_thres:
                left_mask = X[:, feat] <= thres 
                right_mask = ~left_mask
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                if n_left == 0 or n_right == 0:
                    continue 
                gain = parent_gini - (n_left/n) * gini(y[left_mask]) - (n_right/n) * gini(y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat 
                    best_thres = thres 
        return best_feat,  best_thres, best_gain
                
    def build_tree(X, y, depth):
        if depth >= max_depth or len(np.unique(y)) == 1:
            val, counts = np.unique(y, return_counts = True)
            return {'leaf': True,  'label': val[np.argmax(counts)]}

        cur_feat, cur_thres, cur_gain = best_split(X, y)
        if cur_feat is None or cur_gain <= 0:
            val, counts = np.unique(y, return_counts = True)
            return {'leaf': True,  'label': val[np.argmax(counts)]}

        left_mask = X[:, cur_feat] <= cur_thres
        right_mask = ~left_mask

        return {
            'leaf': False, 'feat': cur_feat, 'thres': cur_thres,
            'left': build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def predict_one_tree(node, x_test):
        if node['leaf']:
            return node['label']
        if x_test[node['feat']] <= node['thres']:
            return predict_one_tree(node['left'], x_test)
        else:
            return predict_one_tree(node['right'], x_test)
            
    def predict(lst_trees, X_test):
        votes = [predict_one_tree(tree, X_test) for tree in lst_trees]
        val, counts = np.unique(votes, return_counts = True)
        return val[np.argmax(counts)]

    X_train = np.asarray(X_train)
    n, d = X_train.shape
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    

    lst_trees = []
    for _ in range(n_estimators):
        idx = rn.randint(0, n, size = n)
        tree = build_tree(X_train[idx], y_train[idx], 0)
        lst_trees.append(tree)

    return [predict(lst_trees, x) for x in X_test] 
