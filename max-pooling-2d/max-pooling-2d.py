import numpy as np 
import math 

def max_pooling_2d(X: list, pool_size: int) -> list:
    X = np.array(X)
    output_shape = (int(X.shape[0]/pool_size), int(X.shape[1]/pool_size))
    out = np.empty(output_shape, dtype=np.int32)
    for i in range(0, math.ceil(X.shape[0]/pool_size)):
        for j in range(0, math.ceil(X.shape[1]/pool_size)):
            out[i, j] = np.max(X[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]).astype(np.int32)
    return out.tolist()