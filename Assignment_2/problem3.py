import numpy as np
from matplotlib import pyplot as plt


def cheblog(num): #This is the chebyshev logarithm approximation
    #We get our reference data from 0.5 to 1
    x = np.linspace(0.5,1, 1001)
    y = np.log2(x)
    #We resize our x to be appropriate for chebyshev
    x = np.linspace(-1,1,1001)
    coef = np.polynomial.chebyshev.chebfit(x, y, 100)        

    #We search for an appropriate point to stop
    endpoint = coef.size
    for i,each in enumerate(coef):
        if (abs(each)<=10**(-6)):
            endpoint=i
            break
    #We return the evaluated value
    return np.polynomial.chebyshev.chebval(num, coef[:endpoint])

def mylog2(n):
    #We break down the first number into mantissa and exponent
    n1, n2 = np.frexp(n)
    #We return 
    log2n = n2+cheblog(n1)
    #We could also use the taylor expansion of ln(1+x) centered at x=0, which would simply be 
    # [n**i/i for i in range(appropriate value to approximate to)]
    e1, e2 = np.frexp(np.e)
    log2e = e2+cheblog(e1)
    return log2n/log2e