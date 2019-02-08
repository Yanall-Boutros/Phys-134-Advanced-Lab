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
def chi_squared(ydata, y_bestfit, sigma):
    cs = np.sum(((ydata - y_bestfit)**2)/(sigma**2))
    csr = cs / (len(ydata)-1)
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

def current(xdata, w_0, phi, w, R, V_0, Gamma):
    pass

def w_0(xdata, phi, Gamma):
    # (xdata * tan(\phi) * Gamma - xdata**2)*-1
    return xdata**2 - xdata*np.tan(phi)*Gamma

def phi(xdata, res_freq, Gamma):
    return np.arctan((xdata**2-res_freq**2)/(xdata*Gamma))

def LeastSquaresFit(xdata, ydata, y_sigma, func_pntr, guess_params):
   # Least Squares Fit was originally a python program written by Prof.
   # David Smith, I have adapted and altered it to be generalized as an
   # individual function.
   if type(func_pntr) is not type(LeastSquaresFit):
       print("Function Pointer (func_pntr) not provided.")
       return
   if type(guess_params) is not type([]):
       print("Guess_Params must be type list.")
       return
   xsmooth = np.linspace(np.min(xdata),np.max(xdata), 1000)
   fsmooth = func_pntr(xsmooth, *guess_params)
   popt, pcov = opt.curve_fit(func_pntr, xdata, ydata,
           sigma=y_sigma, p0=guess_params, absolute_sigma=1)
   fsmooth_next = func_pntr(xsmooth, *popt)
   return [xsmooth, fsmooth_next, fsmooth, popt]
