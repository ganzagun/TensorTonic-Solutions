import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    n = y_train.shape[0]
    n_class = np.unique(y_train)

    stats = {}

    for c in n_class:
        y_mask = y_train == c 
        X_mask = X_train[y_mask]
        stats[c] = {
            'prior': np.log(np.sum(y_mask)/ n),
            'mu': np.mean(X_mask, axis = 0),
            'var': np.var(X_mask, axis = 0) + 1e-9
        }

    predictions = []
    for x in X_test:
        best_score = -np.inf
        expected_label = -1
        for c in n_class:
            s = stats[c]
            log_post = s['prior'] + np.sum(
                -(1/2) * np.log(2* np.pi*s['var']) - (x - s['mu'])**2/ (2*s['var'])
            )
            if log_post > best_score:
                best_score = log_post
                expected_label = c 
        predictions.append(expected_label)
       

    return np.array(predictions)        