import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A, dtype = float)
    n, m = A.shape[0], A.shape[1]

    AT = np.zeros((m, n),  dtype = float)

    for i in range(n):
        for j in range(m):
            AT[j][i] = A[i][j]
    return AT
