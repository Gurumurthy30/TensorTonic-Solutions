import math

def normalize_kernel(kernel: list) -> list:
    total = sum(sum(row) for row in kernel)
    return [[value / total for value in row] for row in kernel]

def gaussian_kernel(size: int, sigma: float) -> list:
    ans = [[0] * size for _ in range(size)]
    center = size // 2
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            ans[i][j] = math.exp(-(x**2 + y**2) / (2*sigma**2))  
    normalized_ans = normalize_kernel(ans)          
    return normalized_ans