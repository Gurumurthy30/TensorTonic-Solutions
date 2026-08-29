import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    pointer = 0
    out  = np.zeros((len(v), len(v)))
    for i in range(len(out)):
        for j in range(len(out[0])):
            if i==j:
                out[i][j] = v[pointer]
                pointer+=1
    return out