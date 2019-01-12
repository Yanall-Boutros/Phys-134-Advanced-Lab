# ------------------------------------------------------------------------
# import statements
# ------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math as m
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
import csv
# ------------------------------------------------------------------------
# Function Definitions
# ------------------------------------------------------------------------
def calc_G(T, O):
    # T stands Tau - for the observed oscillation period
    # O looks like Theta - for the equilibrium angle of the balance
    # Constant Definitions
    R = .0449 # meters
    d = 0.06656 # meters
    M = .917 # Kilograms
    return ((((2*pi/T)**2)*R**2)*d/M)*O

def damped_osc(xdata, b=1, a=1, w=1, p=0, m=100):
    # y = e^-bt * a cos(wt-p)
    # b, a, w, and p are the guess parameters
    return np.exp(-1*b*xdata)*a*np.cos(w*xdata-p) + m
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
   param_1 = 0.00104 # b val, controls decay
   param_2 = 60  # a val, controls amp
   param_3 = 0.0296 # freq
   param_4 = np.pi/2 - 0.6 # phase
   param_5 = 101.8 # mean
   params = np.array([param_1, param_2, param_3, param_4, param_5])
   xsmooth = np.linspace(np.min(xdata),np.max(xdata), 1000)
   fsmooth = damped_osc(xsmooth, *params)
   plt.plot(xsmooth, fsmooth, color='green', linewidth=0.5,
           label='Guess', alpha=0.4)
   popt, pcov = opt.curve_fit(damped_osc, xdata, ydata,
           sigma=y_sigma, p0=params, absolute_sigma=1)
   fsmooth_next = damped_osc(xsmooth, *popt)
   plt.plot(xsmooth, fsmooth_next, color='red', linewidth=0.5,
            label='Line of Best Fit', alpha=0.6)
   print("Calculated param vals:")
   print(popt)
   plt.legend(loc=1)
   plt.savefig("LSF.pdf")
# ------------------------------------------------------------------------
# Read in files from directory
# ------------------------------------------------------------------------
with open('data-1.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
# ------------------------------------------------------------------------
# Data Initialization
# ------------------------------------------------------------------------
time = []
pot = []

for i in range(1, len(data)):
    time.append(float(data[i][0]))
    pot.append(float(data[i][1]))

time = np.array(time)
pot = np.array(pot)

# The following slices were found by hand
pot_slice_0 = pot[:218] # Equilibrium one
time_slice_0 = time[:218]

pot_slice_1 = pot[218:2013] # decay 1
time_slice_1 = time[218:2013]

pot_slice_2 = pot[2013:2559] # equil two
time_slice_2 = time[2013:2559]

pot_slice_3 = pot[2559:4400] # decay two
time_slice_3 = time[2559:4400]

pot_slice_4 = pot[4400:] # equil 3
time_slice_4 = time[4400:]
# ------------------------------------------------------------------------
# Data Fitting
# ------------------------------------------------------------------------
# Start with linear fitting on equilibrium points
equil_1 = linfit(
            time_slice_0,
            pot_slice_0,
            np.ones(len(pot_slice_0))
        )

equil_2 = linfit(
            time_slice_2,
            pot_slice_2,
            np.ones(len(pot_slice_2))
        )

equil_3 = linfit(
            time_slice_4,
            pot_slice_4,
            np.ones(len(pot_slice_4))
        )
LeastSquaresFit(time_slice_1, pot_slice_1, np.ones(len(pot_slice_1)))
# ------------------------------------------------------------------------
# Plot Data
# ------------------------------------------------------------------------
plt.errorbar(time, pot, yerr=np.ones(len(pot)), capsize=0.5,
        marker='.', markersize=1, alpha=0.3,
        linestyle='', linewidth=0.5, color='blue', label="data")
plt.legend(loc=1)
plt.xlabel("Time in Seconds")
plt.ylabel("Potential in mV")
plt.savefig("potvtime.pdf")
