import numpy as np
ft = np.fft
from matplotlib import pyplot as plt

def shift(arr, n):
    t = np.linspace(1,len(arr),len(arr))
    G = np.exp(-2*np.pi*t*np.complex(0,1)*n/len(arr)) #We will directly use the convolved form of the shifting array
    Y = ft.fft(y)
    H = G*Y
    h = ft.ifft(H)
    h = -np.real(h) #We correct the sign
    return h
    

#Example code for a gaussian
x = np.linspace(-3,3,1000)
N = 500
y = np.exp(-0.5*x**2) #sigma = 1, amplitude of 1

sh = shift(y,N)
truesh = np.roll(y,N) #This represents the true shift

plt.clf()
l1 = plt.plot(x,sh)
l2 = plt.plot(x,truesh)
l3 = plt.plot(x, truesh-sh)
plt.legend(["FFt","True","Resid"])