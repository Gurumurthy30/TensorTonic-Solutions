import math

def elu(x: list, alpha: float = 1.0) -> list:
    for i in range(len(x)):
        crr = x[i]
        if crr < 0:
            x[i] = alpha * (math.exp(x[i]) - 1)
        
    return x