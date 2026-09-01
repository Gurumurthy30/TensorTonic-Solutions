import numpy as np

def dot_product(x: list, y: list) -> float:
    sum = 0.0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum