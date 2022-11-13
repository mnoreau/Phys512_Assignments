import numpy as np
ft = np.fft
from matplotlib import pyplot as plt

def nowrapconv (f,g) : #The main point I saw online was to bad functions with zeros until they were a power of 2 
    f = npad(f) #So we pad both
    g = npad (g)
    F = ft.fft(f) #Then convolve
    G = ft.fft(g)
    H = F*G
    h = np.real(ft.ifft(H))
    return h/max(h)

def npad (f): #Pad function defined somewhat shakily the while loop could be replaced at some other point
    l = np.size(f)
    t = l
    r= 0.1
    while (r%1 != 0): #We loop until the remainder of the logarithm is a whole number, or increase t
        r = np.log2(t)
        t+=1
    t = int(t-1)
    r = int(t-l)
    temp = np.zeros(t) 
    temp[:l] = f #We insert our array into the first part
    return temp
    

#Plot generation
x = np.linspace(-10,10,50)
y = np.arctan(x)
v = np.sin(x)

res = nowrapconv(y,v)
plt.plot(res)
plt.plot(y)
plt.plot(v)