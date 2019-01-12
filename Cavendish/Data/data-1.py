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

def linfitfunc(xdata, m=50, b=0):
    return m*xdata+b

def LeastSquaresLinFit(xdata, ydata, y_sigma, params, title):
    # Rewritten from LeastSquareFit provided by Prof. David Smith to be
    # adapted for a linear fitting function
   xsmooth = np.linspace(np.min(xdata),np.max(xdata), 1000)
   fsmooth = linfitfunc(xsmooth, *params)
   plt.plot(xsmooth, fsmooth, color='green', linewidth=0.5,
           label='Guess', alpha=0.4)
   popt, pcov = opt.curve_fit(linfitfunc, xdata, ydata,
           sigma=y_sigma, p0=params, absolute_sigma=1)
   fsmooth_next = linfitfunc(xsmooth, *popt)
   plt.plot(xsmooth, fsmooth_next, color='red', linewidth=0.5,
            label='Line of Best Fit', alpha=0.6)
   print("Calculated param vals:")
   print(popt)
   plt.legend(loc=1)
   plt.savefig(title)

def LeastSquaresFit(xdata, ydata, y_sigma, params, title):
   # Least Squares Fit was originally a python program written by Prof.
   # David Smith, I have adapted and altered it to be generalized as an
   # individual function.
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
   plt.savefig(title)
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
# Data Fitting and Plotting
# ------------------------------------------------------------------------
# Guess parameters for equilibrium state one
param_m = 0
param_b = 50
params = np.array([param_m, param_b])
LeastSquaresLinFit(
        time_slice_0,
        pot_slice_0,
        np.ones(len(pot_slice_0)),
        params,
        "Equil_1.pdf"
        )
# Guess parameters for decay one
param_1 = 0.00104 # b val, controls decay
param_2 = 60  # a val, controls amp
param_3 = 0.0296 # freq
param_4 = np.pi/2 - 0.6 # phase
param_5 = 101.8 # mean
params = np.array([param_1, param_2, param_3, param_4, param_5])

LeastSquaresFit(
        time_slice_1,
        pot_slice_1,
        np.ones(len(pot_slice_1)),
        params,
        "Decay 1.pdf"
        )
# Guess parameters for equilibrium point two
param_m = 0
param_b = 101.8
params = np.array([param_m, param_b])
LeastSquaresLinFit(
        time_slice_2,
        pot_slice_2,
        np.ones(len(pot_slice_2)),
        params,
        "Equil_2.pdf"
        )

# Guess parameters for decay two
param_1 = 0.00104
param_2 = -3000
param_3 = 0.0296
param_4 = -1.022 - 0.5
param_5 = 53.8
params = np.array([param_1, param_2, param_3, param_4, param_5])

LeastSquaresFit(
        time_slice_3,
        pot_slice_3,
        np.ones(len(pot_slice_3)),
        params,
        "Decay 2.pdf"
        )
# Guess parameters for equilibrium point three
param_m = 0
param_b = 53.8
params = np.array([param_m, param_b])
LeastSquaresLinFit(
        time_slice_4,
        pot_slice_4,
        np.ones(len(pot_slice_4)),
        params,
        "Equil_3.pdf"
        )
plt.errorbar(time, pot, yerr=np.ones(len(pot)), capsize=0.5,
        marker='.', markersize=1, alpha=0.3,
        linestyle='', linewidth=0.5, color='blue', label="data")
plt.legend(loc=1)
plt.xlabel("Time in Seconds")
plt.ylabel("Potential in mV")
plt.savefig("potvtime.pdf")
