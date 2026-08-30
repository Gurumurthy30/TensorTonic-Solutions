import numpy as np

def swish(x: list) -> np.ndarray:
    x = np.array(x)
    result = x * (1 / (1 + np.exp(-x)))
    return result