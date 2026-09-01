import numpy as np

def dot_product(a: list, b: list) -> float:
    ans = 0.0
    for i in range(len(a)):
        ans += a[i] * b[i]
    return ans
    
def euclidean_norms(a: list) -> float:
    sum_of_sqr = 0.0
    for num in a:
        sum_of_sqr += num ** 2
    return sum_of_sqr ** 0.5

def cosine_similarity(a: list, b: list) -> float:
    dot = dot_product(a,b)
    a_norm = euclidean_norms(a)
    b_norm = euclidean_norms(b)

    if a_norm==0 or b_norm==0:
        return 0.0
        
    return dot/(a_norm * b_norm)