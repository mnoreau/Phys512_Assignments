import numpy as np
from matplotlib import pyplot as plt


n = 100000
y = np.linspace (0,1,n)

def bound (x): #We define our bound 
    if x < 0 :
        return 0
    return np.exp(-x)
    
def gaussbound(x):
    return 1/(np.sqrt(2*np.pi))*np.exp(-0.5*x**2)

def acceptprob(x): #We define the probability distribution of our acceptance function
    if x<np.zeros(len(x)) :
        return 0
    return np.exp(-0.5)*np.exp(-0.5*x**2+x)


def posgauss(s): #we make a function to return positive, gaussian-distributed numbers
    temp = np.zeros(s)    
    for i in range(s):        
        t = np.random.normal(loc=0.0,scale=1.0,size=1)
        if t >=0:
            temp[i]= t
        else:
            i-=1
            continue
    return temp

pred = posgauss(n)

check = np.random.rand(n)

accept = check<pred

yuse = pred[accept]
xuse = gaussbound(yuse)

a,b = np.histogram(xuse,np.linspace(0,1,51))
b_center  = 0.5*(b[1:]+b[:-1])

predic = np.exp(-b_center)

predic = predic/predic.sum()

a = a/a.sum()
sig = np.std(predic-np.exp(-np.linspace(0,1,len(predic)))) #deviates about 0.18

plt.plot(b_center,predic)
plt.bar(b_center, a)
plt.show()