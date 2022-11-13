import numpy as np
from matplotlib import pyplot as plt
N = 100
x = np.linspace(1,51, N)

fy = np.exp(-np.pi*x/(2*N))
win = 0.5-0.5*np.cos(2*np.pi*x/N)

#For plotting the leakage
dft = np.fft.fft(fy*win)
dft = np.fft.fftshift(dft)
plt.plot(np.real(dft))
