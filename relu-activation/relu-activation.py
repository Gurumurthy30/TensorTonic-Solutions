import numpy as np

def relu(x) -> np.ndarray:
    x = np.array(x)
    return np.where(x > 0, x, 0)