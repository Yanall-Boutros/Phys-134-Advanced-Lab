# ========================================================================
# import statements
# ========================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math as m
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
import csv
# ========================================================================
# Function Definitions
# ========================================================================
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
    
    slope, intercept = np.polyfit(xdata, ydata, 1)

    return np.array([slope, intercept, sigma_slope, sigma_intercept])

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

def LeastSquaresFit(xdata, ydata, y_sigma, func, params):
   # Least Squares Fit was originally a python program written by Prof.
   # David Smith, I have adapted and altered it to be generalized as an
   # individual function. I have also reduced the length a decent amount
   # and stripped the plotting aspect (you will have to manually plot
   # your values. fsmooth_next is the best fit y-data. fsmooth is the
   # guessed y-values.
   xsmooth = np.linspace(np.min(xdata),np.max(xdata), len(xdata))
   fsmooth = func(xsmooth, *params)
   popt, pcov = opt.curve_fit(func, xdata, ydata,
           sigma=y_sigma, p0=params, absolute_sigma=1)
   fsmooth_next = damped_osc(xsmooth, *popt)
   plt.legend(loc=1)
   return [xsmooth, fsmooth_next, fsmooth, popt]
# ========================================================================
# Read in files from directory
# ========================================================================
with open('data-1.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
# ========================================================================
# Data Initialization
# ========================================================================
time = []
pot = []

for i in range(1, len(data)):
    time.append(float(data[i][0]))
    pot.append(float(data[i][1]))

time = np.array(time)
pot = np.array(pot)

# The following slices were found by hand. When a new data csv file is
# added, these indiceses will have ot be updated. After they are updated, 
# the rest of the program should operate fine.
pot_slice_0 = pot[:218] # Equilibrium one
time_slice_0 = time[:218]

pot_slice_1 = pot[218:2041] # decay 1
time_slice_1 = time[218:2041]

pot_slice_2 = pot[2041:2555] # equil two
time_slice_2 = time[2041:2555]

pot_slice_3 = pot[2555:4400] # decay two
time_slice_3 = time[2555:4400]

pot_slice_4 = pot[4400:] # equil 3
time_slice_4 = time[4400:]
# ========================================================================
# Data Fitting and Plotting
# ========================================================================
# Equilibrium State One
# ------------------------------------------------------------------------
e1 = linfit(time_slice_0, pot_slice_0, np.ones(len(pot_slice_0)))
e1vals = []
e1vals.append(time_slice_0)
e1vals.append(e1[0]*time_slice_0 + e1[1])
plt.plot(e1vals[0], e1vals[1], color='red', linewidth=0.5,
        label='First Equilibrium Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Decay State One
# ------------------------------------------------------------------------
# quick readjust
dev = time_slice_1[0]
time_slice_1 -= dev*np.ones(len(time_slice_1))
# guess parameters
param_1 = 0.00122 # b val, controls decay
param_2 = -46.76  # a val, controls amp
param_3 = 0.0289 # freq
param_4 = -0.281 # phase
param_5 = 99.68 # mean
params1 = np.array([param_1, param_2, param_3, param_4, param_5])

d1 = LeastSquaresFit(
        time_slice_1,
        pot_slice_1,
        np.ones(len(pot_slice_1)),
        damped_osc,
        params1,
        )
# readjust
d1[0] += dev*np.ones(len(d1[0]))
time_slice_1 += dev*np.ones(len(time_slice_1))
plt.plot(d1[0], d1[1], color='orange',
        linewidth=0.5,
        label='First Decay Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Equilibrium State Two
# ------------------------------------------------------------------------
e2 = linfit(time_slice_2, pot_slice_2, np.ones(len(pot_slice_2)))
e2vals = []
e2vals.append(time_slice_2)
e2vals.append(e2[0]*time_slice_2 + e2[1])
plt.plot(e2vals[0], e2vals[1], color='magenta', linewidth=0.5,
        label='Second Equilibrium Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Decay State Two
# ------------------------------------------------------------------------
# quick readjust for time_slice_3
dev = time_slice_3[0]
time_slice_3 -= dev*np.ones(len(time_slice_3))
# Guess parameters for decay two
param_1 = 0.001203
param_2 = 47.6
param_3 = 0.029
param_4 = -0.319
param_5 = 54.1
params3 = np.array([param_1, param_2, param_3, param_4, param_5])
d2 = LeastSquaresFit(
        time_slice_3,
        pot_slice_3,
        np.ones(len(pot_slice_3)),
        damped_osc,
        params3,
        )
# readjust time_slice_3
d2[0] += dev*np.ones(len(d2[0]))
time_slice_3 += dev*np.ones(len(time_slice_3))
plt.plot(d2[0], d2[1], color='green', linewidth=0.5,
        label='Second Decay Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Equilibrium State Three
# ------------------------------------------------------------------------
e3 = linfit(time_slice_4, pot_slice_4, np.ones(len(pot_slice_4)))
e3vals = []
e3vals.append(time_slice_4)
e3vals.append(e3[0]*time_slice_4 + e3[1])
plt.plot(e3vals[0], e3vals[1], color='cyan', linewidth=0.5,
        label='Third Equilibrium Line of Best Fit', alpha=0.6)
plt.legend(loc=1)
plt.savefig("bestfits.pdf")
# ------------------------------------------------------------------------
# Plot Raw Data with Error Bars
# ------------------------------------------------------------------------
plt.errorbar(time, pot, yerr=np.ones(len(pot)), capsize=0.5,
        marker='.', markersize=1, alpha=0.2,
        linestyle='', linewidth=0.5,
        zorder=10 ,color='blue', label="data")
plt.legend(loc=1, prop={'size': 6})
plt.xlabel("Time in Seconds")
plt.ylabel("Potential in mV")
plt.savefig("potvtime.pdf")
# ========================================================================
# Print Error Analysis
# ========================================================================
# Equilibrium State One
# ------------------------------------------------------------------------
cs = chi_squared(pot_slice_0, e1vals[1], np.ones(len(pot_slice_0)))
print(72*'=')
print("Equilibrium State One Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e1[-2])
print("Intercept Standard Deviation:\t",e1[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# ------------------------------------------------------------------------
# Decay State One
# ------------------------------------------------------------------------
cs = chi_squared(pot_slice_1, d1[1], np.ones(len(pot_slice_1)))
print("Decay State One Error Analysis")
print(72*'=')
print("Decay Value: \t\t\t", d1[-1][0])
print("Amplitude:\t\t\t", d1[-1][1])
print("Frequency:\t\t\t", d1[-1][2])
print("Phase:\t\t\t\t", d1[-1][3])
print("Mean:\t\t\t\t", d1[-1][4])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# ------------------------------------------------------------------------
# Equilibrium State Two
# ------------------------------------------------------------------------
cs = chi_squared(pot_slice_2, e2vals[1], np.ones(len(pot_slice_2)))
print("Equilibrium State Two Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e2[-2])
print("Intercept Standard Deviation:\t",e2[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# ------------------------------------------------------------------------
# Decay State Two
# ------------------------------------------------------------------------
cs = chi_squared(pot_slice_3, d2[1], np.ones(len(pot_slice_3)))
print("Decay State Two Error Analysis")
print(72*'=')
print("Decay Value: \t\t\t", d2[-1][0])
print("Amplitude:\t\t\t", d2[-1][1])
print("Frequency:\t\t\t", d2[-1][2])
print("Phase:\t\t\t\t", d2[-1][3])
print("Mean:\t\t\t\t", d2[-1][4])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# ------------------------------------------------------------------------
# Equilibrium State Two
# ------------------------------------------------------------------------
cs = chi_squared(pot_slice_4, e3vals[1], np.ones(len(pot_slice_4)))
print("Equilibrium State One Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e3[-2])
print("Intercept Standard Deviation:\t",e3[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'=')
