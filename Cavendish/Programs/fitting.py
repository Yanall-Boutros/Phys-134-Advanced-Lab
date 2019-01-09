# ----------------------------------------------------------------------
# import statements
# ----------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math as m
plt.style.use('classic')
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
pi = np.pi
# ----------------------------------------------------------------------
# function definitions
# ----------------------------------------------------------------------
def calc_G(T, O):
    # T stands Tau - for the observed oscillation period
    # O looks like Theta - for the equilibrium angle of the balance
    # Constant Definitions
    R = .0449 # meters
    d = 0.06656 # meters
    M = .917 # Kilograms
    return ((((2*pi/T)**2)*R**2)*d/M)*O

def chi_squared(ydata, y_bestfit, sigma):
    cs = np.sum(((ydata - y_bestfit)**2)/(sigma**2))
    csr = cs / 18
    return (cs, csr)

def W(n, xdata, ydata, yerror):
    w_of_n = np.sum(((xdata**n)*ydata)/(yerror**2))
    return w_of_n

def U(n, xdata, yerror):
    u_of_n = np.sum((xdata**n)/(yerror**2))
    return u_of_n
    

def linfit(xdata, ydata, yerror):
    # output an array of four values in the form
    # (slope, intercept, sigma_slope, sigma_intercept)
    # U_n = \sum [x_i^n divided by \sig_i ^ 2]
    # W_n - \sum [y_i x_i^n divided by \sig_i^2]
    # D = U_0 * U_2 - U_1\^2
    U_0 = U(0, xdata, yerror)
    U_1 = U(1, xdata, yerror)
    U_2 = U(2, xdata, yerror)
    W_0 = W(0, xdata, ydata, yerror)
    W_1 = W(1, xdata, ydata, yerror)
    
    D = U_0*U_2 - (U_1**2)
    slope = ((U_0*W_1) - (U_1*W_0))/D
    intercept = (U_2*W_0 - U_1*W_1)/D
    sigma_slope = (U_0/D)**0.5 # sigma_slope might be squared
    sigma_intercept = (U_2/D)**0.5 # intercept might be squared
    
    return np.array([slope, intercept, sigma_slope, sigma_intercept])

def LeastSquaresFit(xdata, ydata, y_sigma):
   # Least Squares Fit was originally a python program written by Prof.
   # David Smith, I have adapted and altered it to be generalized as an
   # individual function.
   # guess_param_1 = 
   # guess_param_2 = 
   # guess_params = np.array([ ])
   xsmooth = np.linspace(np.min(xdata),np.max(xdata), 1000)
   # fsmooth = GuessFunc(xsmooth, *guess_params)
   plt.plot(xsmooth, fsmooth, color='red',
           label='Guess', alpha=0.9)
   popt, pcov = opt.curve_fit(ImpGuessMagFunc, xdata, ydata,
           sigma=y_sigma, p0=guess_params, absolute_sigma=1)
   fsmooth_next = ImpGuessMagFunc(xsmooth, *popt)
   plt.plot(xsmooth, fsmooth_next, color='red',
            label='Line of Best Fit', alpha=0.5)
   plt.legend(loc=1)
   plt.savefig("LSF.pdf")
