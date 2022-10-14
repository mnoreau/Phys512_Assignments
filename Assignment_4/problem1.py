import numpy as np
from matplotlib import pyplot as plt

data = np.load("sidebands.npz")
t = data['time']
d = data['signal']

#Parameters [a, w, t0]
def lorentz(x, p): 
    l = lambda x,p : p[0]/(1+(x-p[2])**2/p[1]**2) + p[3]/(1+(x-p[2]+p[4])**2/p[1]**2) + p[5]/(1+(x-p[2]-p[4])**2/p[1]**2)
    y = l(x,p)
    
    grad = np.zeros([x.size,p.size])
    for i, each in enumerate(p): #We iterate through all parameters and do a numerical derivative of them
        p[i]+=1e-6
        grad[:,i] = (l(x,p)-y)/1e-6
        p[i]=each #We reset the parameter value
    
    return y,grad
#This is a fairly bad guess, but necessary for now as it will otherwise output a singular matrix, which can't be inverted
par = np.array([1.4,1.5,1.5,1,0.01,1]) #[a,w,t0,b,dt,c]

for j in range(5): #This is fairly standard, we calculate our prediction and gradient
    pred, gradient = lorentz(t,par)
    r = d-pred #We get our residuals
    err = (r**2).sum()
    r = np.matrix(r).transpose()
    gradient = np.matrix(gradient)
    #We do the matrix multiplication
    lhs = gradient.transpose()@gradient
    rhs = gradient.transpose()@r
    dp = np.linalg.inv(lhs)@(rhs)
    
    for jj in range(par.size): #We update our guess
        par[jj] = par[jj] + dp[jj]

# for jj in range(par.size): #We add a perturbation above
#     par[jj] = par[jj]+dp[jj]
# perturbation, grad = lorentz(t, par)
# #We add a perturbation below
# for jj in range(par.size):
#     par[jj] = par[jj]-2.0*dp[jj]
# invpert, grad = lorentz(t,par)

resid = abs(d-pred)
mean_resi = np.mean(resid)
var_resi = np.std(resid, ddof=1)

# plt.plot(t,pred)
# plt.plot(t,perturbation)
# plt.plot(t,invpert)
# plt.legend(["Prediction","Perturbation", "Negative perturbation"])

# plt.plot(t,resid)
# plt.axhline(mean_resi)
# plt.legend(["Residuals","Mean"])

# plt.plot(t, pred)
# plt.plot(t,d)
# plt.legend(["Pred","True"])