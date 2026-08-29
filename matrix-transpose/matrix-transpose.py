import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    output = np.empty((len(A[0]), len(A)))
    for i in range(len(A)):
        for j in range(len(A[0])):
            output[j][i] = A[i][j]
    return output    