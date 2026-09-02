import numpy as np

def expected_value_discrete(x: list, p: list) -> float:
    n = len(x)
    total = 0.0
    for i in range(0, n):
        total += x[i] * p[i]

    return total