import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    x = np.array(x)
    return np.where(x > 0, x, alpha * x)