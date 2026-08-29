import numpy as np

def matrix_trace(A: list) -> float:
    out = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if i == j:
                out += A[i][j]
    return float(out)