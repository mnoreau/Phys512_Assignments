import numpy as np
from matplotlib import pyplot as plt

#This is the normal runge-kutta
def r4k_step(fun, x, y, h):
    k1 = fun(x,y)
    k2 = fun(x+h/2,y+k1*h/2)
    k3 = fun(x+h/2, y+k2*h/2)
    k4 = fun(x+h, k3*h)
    
    return y+(h/6.0*(k1+2*k2+2*k3+k4))

#We use the runge-kutta to integrate over some amount of points
def integrator(f,start, stop, init, n):
    step = (stop-start)/n
    res = np.zeros(n)
    absc = start
    res[0] = init
    for i, each in enumerate(res[1:], 1):
        res[i] = r4k_step(f, absc, init, step)
        #We update our guess
        absc+=step
        init = res[i]

    return res

#This is the runge-kutta again, but with the Delta added
def rk4_stepd(fun, x0, y0, h):
    temp = 0
    minitemp = 0
    
    temp = r4k_step(fun, x0, y0, h)
    minitemp = r4k_step(fun, x0, y0, h/2)
    minitemp = r4k_step(fun, x0+h/2, minitemp, h/2)
    
    return minitemp + (minitemp-temp)/15

#This integrates like integrator(), but I using rk4_stepd
def dintegrator(f, start, stop, init, n):
    step = (stop-start)/n
    res = np.zeros(n)
    absc = start
    res[0] = init
    for i, each in enumerate(res[1:], 1):
        res[i] = rk4_stepd(f, absc, init,step)
        absc += step
        init = res[i]
    return res

diff = lambda x,y :  y/(1+x**2)
true = lambda x:4.57605801029808*np.exp(np.arctan(x))
evalspace1 = np.linspace(-20, 20, 200)
evalspace2 = np.linspace(-20,20,66)

pred1 = integrator(diff, -20, 20, 1, 200)
pred2 = dintegrator(diff, -20, 20, 1, 66)

real1 = true(evalspace1)
real2 = true(evalspace2)


plt.ion()
plt.plot(evalspace1, real1)
plt.plot(evalspace1, pred1)
plt.plot(evalspace1, pred1-real1)
plt.legend(["True values","Prediction","Residuals"])
plt.title("Normal Runge-Kutta")
