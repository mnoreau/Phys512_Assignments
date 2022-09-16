import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

dat = np.loadtxt('lakeshore.txt')

def lakeshore(V, data):
#Variable initialisations
        V = np.array(V)
        
        temp=data[:,0] 
        voltage = data[:,1]
        diff = data[:,2]
        length=np.size(V)
        result=np.zeros(length)
        errors=np.zeros(length)
        erroronerror=np.zeros(length)
        #We compute our spline
        spline = interpolate.CubicSpline(temp, voltage)
        i=0
        if(length==1): #Case where the array is 0-dimensional
                result[i]=spline(V) #This is the main interpolation result
                errors[i],erroronerror[i] = errorcalc(V, temp, voltage)#This is an error estimate
                return result[0], errors[0]
        for each in V:  #We iterate over all of our voltages if the input is biffer
                result[i]=spline(each) 
                errors[i],erroronerror[i] = errorcalc(each, temp, voltage) 
        return result,errors
        
#This is in the most part taken from the bootstrap_interp.py file created in the tutorial
def errorcalc(x,xdat, ydat):
        rng = np.random.default_rng()
        generated = np.zeros(3)
        #We'll resample 3 times, since we have 144 data points, we 
        for index in range(0,3):
                indices = list(range(xdat.size))
                interpol = rng.choice(indices, size = int(144/3), replace=False)
                interpol.sort()
                new_interpol = interpolate.CubicSpline(xdat[interpol], ydat[interpol])
                generated[index] = new_interpol(x)

        stds=np.std(generated, ddof=1)
        error2 = np.mean(stds)
        error2_std = np.std(stds)
        return error2, error2_std
