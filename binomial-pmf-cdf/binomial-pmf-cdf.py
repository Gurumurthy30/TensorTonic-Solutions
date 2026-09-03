import math

def _pmf(n: int, p: float, k: int) -> int:
    return math.comb(n,k) * (p ** k) * (1 - p)**(n-k)

def _cdf(n: int, p: float, k: int) -> int:
    cdf = 0
    for i in range(k + 1):
        cdf += (math.comb(n, i) * (p**i) * (1-p)**(n-i))
    return cdf

def binomial_pmf_cdf(n: int, p: float, k: int) -> dict:
    pmf = _pmf(n,p,k)
    cdf = _cdf(n,p,k)
    return  {"pmf": float(pmf), "cdf": float(cdf)}