import math

def selu(x: list) -> list:
    Lambda = 1.0507
    alpha = 1.6733
    
    for i in range(len(x)):
        crr = x[i]
        if crr < 0:
            x[i] = Lambda * alpha * (math.exp(x[i]) - 1)
        else:
            x[i] = Lambda * x[i]
        
    return x