import numpy as np
ft = np.fft
from matplotlib import pyplot as plt

def corrshift (y, nshift): #We use our new definition of the correlation function
    g = np.roll(y,int(nshift*len(y))) #We use the easily accessible numpy function for shifting
    corr = ft.ifft(ft.rfft(y)*np.conj(ft.rfft(g)))
    return np.real(corr)


#This is just the means of generating the example, we iterate through a variety of shift-proportions from 0/1 to 1/1
x = np.linspace(-5,5,1000)
f = np.exp(-x**2/6)
p = list()
for i in np.linspace(0,1,11):
    # p.append() = corrshift(f,i)
    plt.plot(corrshift(f,i))
    p.append(f"Shift of {i:.1f}")

plt.legend(p)