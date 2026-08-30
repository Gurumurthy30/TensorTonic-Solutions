import numpy as np

def sigmoid(x):
    x = np.array(x)
    x = 1 / (1 + np.exp(-x))
    return x