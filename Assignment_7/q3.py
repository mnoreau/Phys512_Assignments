import numpy as np
from matplotlib import pyplot as plt
n=100000

u = np.random.uniform(0.00000000001,1.0,n) #So there is no division by 0
v = np.random.uniform(0,np.exp(-1),n)
vdivu = list()
for i in range(n):
    if (u[i] <= np.sqrt(np.exp(-v[i]/u[i]))):
        vdivu.append(v[i]/u[i])

plt.hist(vdivu,bins=np.linspace(0,1,41))
sig = np.std(vdivu-np.exp(-np.linspace(0,1,len(vdivu)))) #Deviates from about 0.96