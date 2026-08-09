import numpy as np

def sigmoid(x):
    np_x = np.array(x)
    sig = 1/(1 + np.exp(- np_x))
    return sig