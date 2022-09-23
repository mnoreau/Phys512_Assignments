import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as it
from functools import partial
#We take our integrator from problem 2, but we should be careful not to have divisions by 0 inside

def integrate_adaptive(fun, a, b, tol, extra=None):
        #The concept of this is just taken from the Adaptive Simpson's method wikipedia article at https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method
        if(extra==None):
        #This is the initial run, which establishes a midpoint, and evaluates the endpoints
                tol = abs(tol) #Fixes case of bad tol that kept hitting max recursion depth
                #global count
                #count=2
                fa, fb = fun(a),fun(b)
                mid, fmid, whole = simpsons(fun, a, fa, b, fb)
                return integrate_adaptive(fun, a, b, tol, [fa, fb, whole, mid, fmid]) #We will then run the program again, but with the new information
        else:
        #We split our integral in two and evaluate it
                fa, fb, whole, mid, fmid = extra[0],extra[1],extra[2],extra[3],extra[4]
                lmid, flmid, left = simpsons(fun, a, fa, mid, fmid) #simpsons on the left side
                rmid, frmid, right = simpsons(fun, mid, fmid, b, fb) #simpsons on the right side
                err = left+right-whole
                if abs(err)<=tol*15: #If we have reached under our error, we will return the integral
                        return left+right+err/15 
                #Otherwise, we start again our process, replacing our midpoint with quarterpoints
                return integrate_adaptive(fun,a,mid,tol/2, [fa,fmid, left, lmid, flmid]) + integrate_adaptive(fun,mid,b, tol/2, [fmid,fb, right, rmid, frmid])

def simpsons(func, a, fa, b, fb): #This is the function that does the heavy lifting, it is our simpson's method and midpoint calculator
        m = (a+b)/2
        fm = func(m)
        #global count
        #count+=1
        return m, fm, (abs(b-a)/6*(fa+4*fm+fb))


#We take values of 
R=1
z=0
def fun (u):
    #print((R**2+z**2-2*R*u))
        return (z-u*R)/np.power((R**2+z**2-2*R*u),(3/2))

xdat = np.linspace(0,2, 50)
xdatl = np.linspace(0,0.999,25)
xdatr = np.linspace(1, 2, 25)
mypred = np.zeros(xdat.size)

# for i, each in enumerate(xdatl):
#     z=each #We will integrate the value over one z
#     mypred[i] = integrate_adaptive(fun, -1, -0.01, 10**(-6))

# for i, each in enumerate(xdatr):
#     z=each #We will integrate the value over one z
#     mypred[xdatl.size+i] = integrate_adaptive(fun, -1, -0.01, 10**(-6))
scipred = np.zeros(xdat.size)
for i, each in enumerate(xdatl):
    z = each
    scipred[i] = it.quad(fun, -1, 1,full_output=1, epsabs=10**(-6))[0]
    if (np.isnan(scipred[i])):
        scipred[i]=0

for i, each in enumerate(xdatr):
    z=each
    scipred[xdatl.size+i] = it.quad(fun,-1, 1, full_output=1, epsabs=10**(-6))[0]
    # if (np.isnan(scipred[xdatl.size+i])):
    #     scipred[xdatl.size+i]=0

#plt.plot(xdat, pred)
plt.plot(xdat, scipred)
# plt.plot(xdat,mypred)
plt.xlabel("z")
plt.ylabel("Ez*(2*eps_0)/(R*sigma)")
