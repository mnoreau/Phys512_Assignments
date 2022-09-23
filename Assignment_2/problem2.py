import numpy as np 
from scipy import interpolate
from matplotlib import pyplot as plt


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
                if abs(err)<=tol: #If we have reached under our error, we will return the integral
                        return left+right+err/15 
                #Otherwise, we start again our process, replacing our midpoint with quarterpoints
                return integrate_adaptive(fun,a,mid,tol/2, [fa,fmid, left, lmid, flmid]) + integrate_adaptive(fun,mid,b, tol/2, [fmid,fb, right, rmid, frmid])

def simpsons(func, a, fa, b, fb): #This is the function that does the heavy lifting, it is our simpson's method and midpoint calculator
        m = (a+b)/2
        fm = func(m)
        #global count
        #count+=1
        return m, fm, (abs(b-a)/6*(fa+4*fm+fb))

#f = np.exp
#x = np.linspace(0,10,101)
#y = f(x)
#pred = integrate_adaptive(f, -1,1,10**(-6))
#print(f"We predict {pred} with {count} function calls")
