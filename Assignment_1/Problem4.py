import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

fun1 = np.cos
fun2 = lambda x: 1/(1+x**2)

#We will use 15 points as our inputs
datapoints1 = np.linspace(-np.pi/2, np.pi/2, 4)
datapoints2 = np.linspace(-1, 1, 14)
#The functions will all be tested over 101 points
evalpoints1 = np.linspace(-np.pi/2, np.pi/2, 101)
evalpoints2 = np.linspace(-1,1,101)


#First, we will use the polynomial fit
#In order to give polyfit the best chances, we will give it an even order number since we have symmetric functions
polycoeffs = np.polyfit(datapoints1, fun1(datapoints1), len(datapoints1)-1)
polyresult = np.polyval(polycoeffs, evalpoints1)
#We then evaluate our error for the fit
polyerror = abs(polyresult-fun1(evalpoints1))
#Finally, we evaluate a mean error on the fit
polyerror1 = (np.mean(polyerror), np.std(polyerror, ddof=1))


#Then, we will make the spline calculations
spline = interpolate.splrep(datapoints1, fun1(datapoints1))
splineresult = interpolate.splev(evalpoints1, spline)

splinerror = abs(splineresult-fun1(evalpoints1))
splinerror1 = (np.mean(splinerror), np.std(splinerror, ddof=1))

#Finally, we will calculate the rational polynomial function, for which the code is greatly inspire by the one seen in class:
orderp = 1
orderq = 2

#We create the polynomials
pcols = [datapoints1**k for k in range(orderp+1)]
p = np.vstack(pcols)

qcols = [-datapoints1**k*fun1(datapoints1) for k in range(1,orderq+1)]
q = np.vstack(qcols)

matrix = np.hstack([p.T,q.T])

#We calculate the appropriate coefficients
coeffs = np.linalg.inv(matrix)@fun1(datapoints1)

resultp = 0 
#We evaluate the function at the evaluation points
for i in range(orderp+1):
	resultp += coeffs[i]*evalpoints1**i
resultq = 1
for i in range(orderq):
	resultq += coeffs[orderp+1+i]*evalpoints1**i

#We calculate the result and errors
ratioresult = resultp/resultq

ratioerror1 = abs(ratioresult-fun1(evalpoints1))
ratioerror1 = (np.mean(ratioerror1), np.std(ratioerror1, ddof=1))

##print("Polynome", polyerror1)
##print("Spline", splinerror1)
##print("Rational", ratioerror1)

##plt.clf()
##plt.plot(evalpoints1, polyresult)
##plt.plot(evalpoints1, splineresult)
##plt.plot(evalpoints1,ratioresult)
##plt.plot(evalpoints1, fun1(evalpoints1))
##plt.legend(["Polynomial", "Spline", "Rational", "True"])
##plt.show()

#This next section is the same as the previous one, but with the lorentzian function and appropriate data/evaluation points
#First, we will use the polynomial fit
#In order to give polyfit the best chances, we will give it an even order number since we have symmetric functions
polycoeffs = np.polyfit(datapoints2, fun2(datapoints2), len(datapoints2)-1)
polyresult = np.polyval(polycoeffs, evalpoints2)
#We then evaluate our error for the fit
polyerror = abs(polyresult-fun2(evalpoints2))
#Finally, we evaluate a mean error on the fit
polyerror2 = (np.mean(polyerror), np.std(polyerror, ddof=1))


#Then, we will make the spline calculations
spline = interpolate.splrep(datapoints2, fun2(datapoints2))
splineresult = interpolate.splev(evalpoints2, spline)

splinerror = abs(splineresult-fun2(evalpoints2))
splinerror2 = (np.mean(splinerror), np.std(splinerror, ddof=1))

#Finally, we will calculate the rational polynomial function, for which the code is greatly inspire by the one seen in class:
orderp = 6
orderq = 7

#We create the polynomials
pcols = [datapoints2**k for k in range(orderp+1)]
p = np.vstack(pcols)

qcols = [-datapoints2**k*fun2(datapoints2) for k in range(1,orderq+1)]
q = np.vstack(qcols)

matrix = np.hstack([p.T,q.T])

#We calculate the appropriate coefficients
coeffs = np.linalg.inv(matrix)@fun2(datapoints2)

resultp = 0 
#We evaluate the function at the evaluation points
for i in range(orderp+1):
	resultp += coeffs[i]*evalpoints1**i
resultq = 1
for i in range(orderq):
	resultq += coeffs[orderp+1+i]*evalpoints2**i

#We calculate the result and errors
ratioresult = resultp/resultq

ratioerror2 = abs(ratioresult-fun2(evalpoints2))
ratioerror2 = (np.mean(ratioerror2), np.std(ratioerror2, ddof=1))

print("Polynome", polyerror2)
print("Spline", splinerror2)
print("Rational", ratioerror2)

plt.clf()
##plt.plot(evalpoints2, polyresult)
##plt.plot(evalpoints2, splineresult)
plt.plot(evalpoints2,ratioresult)
plt.plot(evalpoints2, fun2(evalpoints2))
##plt.legend(["Polynomial", "Spline", "True"])
plt.show()

