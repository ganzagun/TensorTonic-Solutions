import numpy as np

def dbscan(X, eps=0.5, min_samples=5):
    """
    Returns: list of integer labels (-1 for noise)
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    labels = np.full(n, -1)
    cluster_id = 0

    def region_query(idx):
        dists = np.sqrt(np.sum((X - X[idx])**2, axis=1))
        return list(np.where(dists <= eps)[0])

    visited = set()

    for i in range(n):
        if i in visited:
            continue
        visited.add(i)

        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = set(neighbors) - {i}
        seed_list = list(seed_set)

        j = 0
        while j < len(seed_list):
            q = seed_list[j]
            if q not in visited:
                visited.add(q)
                q_neighbors = region_query(q)
                if len(q_neighbors) >= min_samples:
                    for nb in q_neighbors:
                        if nb not in seed_set:
                            seed_set.add(nb)
                            seed_list.append(nb)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels.tolist()
