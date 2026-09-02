import numpy as np

def bernoulli_pmf_and_moments(x: list, p: float) -> dict:
    success = p
    failure = 1 - p
    # PMF list
    for i in range(len(x)):
        crr = x[i]
        if crr==0:
            x[i] = failure
        else:
            x[i] = success

    # var
    var = float(p*(1-p))
    
    return  {"pmf": np.array(x), "mean": float(p), "variance": var} 