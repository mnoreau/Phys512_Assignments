import numpy as np
import camb
from matplotlib import pyplot as plt


planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3]);

def get_spectrum(garb,pars,lmax=3000): #I had functions defined for f(x,par) so I just added a placeholder
#variable to get it into shape. Other wise, it is essentially the same as the prof version
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:]


# def gauss (x,p): 
#     A,s = p[0],p[1]
#     return A*np.exp(-x**2/(2*s**2))

#Differentiator
def dif (f,x,p): #Takes in the input funtion, the independant value and list of parameters and outputs a derivative
    grad = np.zeros((x.size,p.size))
    #We take the gradient for each parameter of the function
    for i, each in enumerate(p): #We differentiate each parameter accordingly
        p[i]= p[i]+(1e-6) #If we're going to make two function calls anyway, might as well differentiate from both sides
        forward = f(x,p)
        p[i]=p[i]-(2e-6)
        backward = f(x,p) 
        grad[:,i] = (forward-backward)/2e-6
        p[i] = each
    return grad

def newt(f,x,y,init,N): #Newton's method
    for j in range(N): #We do a certain amount of steps
        pred = f(x,init) #Classical, we get our initial prediction and gradient
        grad = dif(f,x,init)
        r = y-pred #Get the residuals and error
        err = (r**2).sum()
        
        u,s,vt = np.linalg.svd(grad,False) #SVD to finally get rid of the singular matrix error
        s = np.matrix(np.diagflat(1.0/s)) #Can't believe it took me so long to remember it
        d = np.matrix(x).transpose()
        dp = vt.transpose()*s*u.transpose()*d
        for jj in range(init.size):
            init[jj] = init[jj]+dp[jj]
    parerr = np.linalg.inv(vt.transpose()*s*s*vt.transpose()) #A decomposed (At * A)^-1
    pred = f(x,init)
    grad = dif(f,x,init)
    err = ((y-pred)**2).sum()
    return pred, init, err, parerr

def mc(f,x,y,par,parerr,err,nstep,nchain):
    chains = [None]*nchain
    for i in range(nchain):
        chain = np.zeros([nstep,len(p0)])
        error = np.zeros([nstep,len(p0)])
        chain[0,:] = par+3.0*np.random.randn(len(par))*parerr
        for j in range(1,nstep):
            pp = chain[j-1,:]+1.0*np.random.randn(len(parerr))*parerr
            newerr = np.sum((f(x,par)-y))**2
            accept = np.exp(-0.5*(newerr-err))
            if accept > np.random.rand(1):
                chain[j,:]=pp
                error[j,:]=newerr
            else:
                chain[j,:]=chain[j-1,:]
        chains[i]=chain
    return chains

p0 = np.asarray([68,0.022,0.12,0.055,2.10e-9,1.1])


pred, par, err, parerr = newt(get_spectrum,np.zeros(3049),spec,p0,5)
print(par)
plt.plot(err)
#np.savetxt("planck_fit_params.txt", np.hstack(par,parerr), header='Parameters Parameter_errors')
