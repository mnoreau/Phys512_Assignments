import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

#We have the list of half-lives of all the elements in the decomposition chain as [U238, Th234, Pa234, U234, Th230, Ra226, Rn222, Po218, Pb214, Bi214, Po214, Pb210, Bi210, Po210, Pb206]
def fun(x,y, half_life=[4.468e9*365.25*24.0*3600.0, 24.10*24.0*3600.0, 6.70*3600, 245500.0*365.0*24.0*3600.0, 75380.0*365.0*24.0*3600.0,1600.0*365.0*24.0*3600.0,3.8235*24.0*3600.0,3.1*60.0, 26.8*60.0,19.9*60.0,164.3e-6,22.3*365.0*24.0*3600.0,5.015*365.0*3600.0,138.376*24.0*3600.0]):
    dydx = np.zeros(len(half_life)+1)
    dydx[0] = -y[0]/half_life[1]
    for i in range(1, len(half_life)):
        dydx[i] = y[i-1]/half_life[0]-y[i]/half_life[i]
        dydx[i+1] = y[i]/half_life[i]
    return dydx

#We initialize our problem with 
y0 = np.zeros(15)
y0[0] = 1
x0 = 0
x1 = 1e10
ans = integrate.solve_ivp(fun, [x0,x1], y0, method="Radau")

plt.ion()
absc = np.linspace(0,1e10, ans.y[0].size)
plt.plot(absc, ans.y[-1]/ans.y[0] )