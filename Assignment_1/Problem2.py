import numpy as np
from matplotlib import pyplot as plt


#Here is the main function
def ndiff(fun, x, full=False):
    
    length = np.size(x)
    #We implement lists in case x is a list
    errorlist = np.zeros(length)
    derivlist = np.zeros(length)
    steplist = np.zeros(length)
    errorlist[0] = 10
    
    i=0
    if length==1: #Case where x is simply as value
        while errorlist[i] >=10**(-7) and dx>10**(-16): #We decide arbitrarily that our tolerance will be 10^(-7)
            dx = dx/2 #This step size reduction could probably be optimized
            errorlist[i] = errorcal(fun, x dx) #We compute the error to see if we need a better step
        derivlist[i]=centraldiff(fun,x,dx)
        if(full):
            return derivlist[0], steplist[0],errorlist[0] 
        return derivlist[0]
            
    for each in x: #This allows us to iterate over x if it is a list
        dx = 2e-4 #We start with a considerably large step

        #This is the loop that will help us refine our step size
        while errorlist[i] >=10**(-7) and dx>10**(-16): #We decide arbitrarily that our tolerance will be 10^(-7)
            dx = dx/2 #This step size reduction could probably be optimized
            errorlist[i] = errorcal(fun, each, dx) #We compute the error to see if we need a better step

        #We take note of our final results
        derivlist[i] = centraldiff(fun, each, dx)
        steplist[i] = dx
        #We continue to a new loop and increment the index
        i+=1
    if full:
        #If we have a list input
        return derivlist, steplist, errorlist
    #This only runs if full is False
    return derivlist # we return only the derivatives

    
def centraldiff(fun, x, step):
    return (fun(x+step)-fun(x-step))/(2*step)

#We calculate the double derivative for the error calculation
def doublediff(fun,x,step):
    return (centraldiff(fun,x+step,step)-centraldiff(fun,x-step,step))/(2*step)

#This function approximates the error using the formula seen in class of epsilon = dx^3*f''/f
def errorcal(fun, x, step):
    return step**3*doublediff(fun,x,step)/fun(x)
