import numpy as np
from matplotlib import pyplot as plt

def r4k_step(fun, x, y, h):
    k1 = fun(x,y)
    k2 = fun(x+h/2,y+k1*h/2)
    k3 = fun(x+h/2, y+k2*h/2)
    k4 = fun(x+h, k3*h)
    
    return y+(h/6.0*(k1+2*k2+2*k3+k4))

def integrator(f,start, stop, init, n):
    step = (stop-start)/n
    res = np.zeros(n)
    absc = start
    res[0] = init
    for i, each in enumerate(res[1:], 1):
        res[i] = r4k_step(f, absc, init, step)

    return res

def stepd(fun, x0, y0, h):
    temp = 0
    minitemp = 0
    
    temp = r4k_step(f, absc, init, step)
    minitemp = r4k_step(f, absc, init, step/2)
    minitemp = r4k_step(f, absc+step/2, minitemp, step/2)
    return minitemp + (minitemp-temp)/15

def dintegrator(f, start, stop, init, n):
    step = (stop-start)/n
    res = np.zeros(n)
    absc = start
    res[0] = init
    for i, each in enumerate(res[1:], 1):
        res[i] = r4k_stepd(f, absc, init,step)
    return res