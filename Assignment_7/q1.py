from matplotlib import pyplot as plt
import numpy as np
import random

def get_rands_nb(vals):
    n=len(vals)
    for i in range(n):
        vals[i]=random.random()
    return vals

def get_rands(n):
    vec=np.empty(n,dtype='float64')
    get_rands_nb(vec)
    return vec


n=30000
data=get_rands(n*3)
#vv=vec&(2**16-1)

dd=np.reshape(data,[n,3])
dmax=np.max(dd,axis=1)

maxval=1
dd2=dd[dmax<maxval,:]

# f=open('rand_points.txt','w')
# for i in range(vv2.shape[0]):
#     myline=repr(vv2[i,0])+' '+repr(vv2[i,1])+' '+ repr(vv2[i,2])+'\n'
#     f.write(myline)
# f.close()


data = dd2.T

# data = np.loadtxt('rand_points.txt').T
x = data[0]
y = data[1]
z = data[2]

sub,ax = plt.subplots(subplot_kw = {"projection":"3d"})
ax.scatter(x,y,z,marker=".")
plt.show()