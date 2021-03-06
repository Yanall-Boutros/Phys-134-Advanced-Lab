# =======================================================================
# import statements
# =======================================================================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math as m
import numpy.random as ran
import scipy.optimize as opt
import scipy.stats as stat
import csv
# =======================================================================
# Function Definitions
# =======================================================================
def determine_sigmas(yvals, ybestfit, return_sigma):
    for j in range(5):
        rcs = chi_squared(yvals, ybestfit, return_sigma)[1]
        # for every error in return_sigma, if the change lowers the
        # reduced
        # chi square, save the change to return sigma.
        for i in range(len(return_sigma)):
            rcs = chi_squared(yvals, ybestfit, return_sigma)[1]
            copy = return_sigma[:]
            copy[i] *= 1.1
            ccs = chi_squared(yvals, ybestfit, copy)[1]
            if rcs > ccs and ccs >= 2:
                # If the copied version is lower, save the changes, and
                # rerun
                return_sigma = copy.copy()
        # return the final result
    return return_sigma, rcs, ccs
def diff_G(G, diff):
    gact = 6.67*10**-11
    gours = (G - gact)/(gact)
    gerrp = (G - gact + diff)/gact
    gerrn = (G - gact - diff)/gact
    return gours, gerrp, gerrn 
def calc_O(defl):
    L = 2.593
    return defl/(4*L)
def calc_K(T):
    d = 0.06656 # meters
    m = .0147 # Kilograms
    return 2*m*d**2*(2*np.pi/T)**2
def calc_G(T, O):
    # T stands Tau - for the observed oscillation period
    # O looks like Theta - for the equilibrium angle of the balance
    # Constant Definitions
    R = .0449 # meters
    d = 0.06656 # meters
    M = .917 # Kilograms
    return ((((2*np.pi/T)**2)*R**2)*d/M)*O

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
# =======================================================================
# Read in files from directory
# =======================================================================
with open('goodrun.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
# =======================================================================
# Data Initialization
# =======================================================================
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
pot_slice_0 = pot[:100] # Equilibrium one
time_slice_0 = time[:100]

pot_slice_1 = pot[100:2000] # decay 1
time_slice_1 = time[100:2000]

pot_slice_2 = pot[2000:2625] # equil two
time_slice_2 = time[2000:2625]

pot_slice_3 = pot[2625:4750] # decay two
time_slice_3 = time[2625:4750]

pot_slice_4 = pot[4750:] # equil 3
time_slice_4 = time[4750:]
# =======================================================================
# Data Fitting and Plotting
# =======================================================================
# Equilibrium State One
# -----------------------------------------------------------------------
e1 = linfit(time_slice_0, pot_slice_0, np.ones(len(pot_slice_0)))
e1vals = []
e1vals.append(time_slice_0)
e1vals.append(e1[0]*time_slice_0 + e1[1])
plt.plot(e1vals[0], e1vals[1], color='red', linewidth=0.5,
        label='First Equilibrium Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Run error bar fix
OG_ERRS = np.array([])
rcs = 1.5
ccs = 1
return_sigma_0 = np.ones(len(pot_slice_0))
while (rcs/ccs > 1.49):
    return_sigma_0, rcs, ccs = determine_sigmas(
            pot_slice_0,
            e1vals[1],
            return_sigma_0)
OG_ERRS = np.append(OG_ERRS, return_sigma_0)
# -----------------------------------------------------------------------
# Decay State One
# -----------------------------------------------------------------------
# quick readjust
dev = time_slice_1[0]
time_slice_1 -= dev*np.ones(len(time_slice_1))
# guess parameters
param_1 = 0.0013 # b val, controls decay
param_2 = -46.51  # a val, controls amp
param_3 = 0.0291 # freq
param_4 = 0.0408 # phase
param_5 = 101.1 # mean
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
# ------------------------------------------------------------------------
# Run error bar fix
rcs = 2
ccs = 1
return_sigma_1 = np.ones(len(pot_slice_1))
while (rcs/ccs > 1.1):
    return_sigma_1, rcs, ccs = determine_sigmas(
            pot_slice_1,
            d1[1],
            return_sigma_1)
OG_ERRS = np.append(OG_ERRS, return_sigma_1)
# ------------------------------------------------------------------------
plt.plot(d1[0], d1[1], color='orange',
        linewidth=0.5,
        label='First Decay Line of Best Fit', alpha=0.6)
# -----------------------------------------------------------------------
# Equilibrium State Two
# -----------------------------------------------------------------------
e2 = linfit(time_slice_2, pot_slice_2, np.ones(len(pot_slice_2)))
e2vals = []
e2vals.append(time_slice_2)
e2vals.append(e2[0]*time_slice_2 + e2[1])
plt.plot(e2vals[0], e2vals[1], color='magenta', linewidth=0.5,
        label='Second Equilibrium Line of Best Fit', alpha=0.6)
# ------------------------------------------------------------------------
# Run error bar fix
rcs = 2
ccs = 1
return_sigma_2 = np.ones(len(pot_slice_2))
while (rcs/ccs > 1.1):
    return_sigma_2, rcs, ccs = determine_sigmas(
            pot_slice_2,
            e2vals[1],
            return_sigma_2)
OG_ERRS = np.append(OG_ERRS, return_sigma_2)
# -----------------------------------------------------------------------
# Decay State Two
# -----------------------------------------------------------------------
# quick readjust for time_slice_3
dev = time_slice_3[0]
time_slice_3 -= dev*np.ones(len(time_slice_3))
# Guess parameters for decay two
param_1 = 0.001142
param_2 = 43.55
param_3 = 0.029
param_4 = -0.203
param_5 = 55.8
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
# Run error bar fix
rcs = 2
ccs = 1
return_sigma_3 = np.ones(len(pot_slice_3))
while (rcs/ccs > 1.1):
    return_sigma_3, rcs, ccs = determine_sigmas(
            pot_slice_3,
            d2[1],
            return_sigma_3)
OG_ERRS = np.append(OG_ERRS, return_sigma_3)
# -----------------------------------------------------------------------
# Equilibrium State Three
# -----------------------------------------------------------------------
e3 = linfit(time_slice_4, pot_slice_4, np.ones(len(pot_slice_4)))
e3vals = []
e3vals.append(time_slice_4)
e3vals.append(e3[0]*time_slice_4 + e3[1])
plt.plot(e3vals[0], e3vals[1], color='cyan', linewidth=0.5,
        label='Third Equilibrium Line of Best Fit', alpha=0.6)
plt.legend(loc=1, prop={'size': 6})
plt.xlabel("Time in Seconds")
plt.ylabel("Potential in mV")
plt.savefig("rgbestfits.pdf")
# ------------------------------------------------------------------------
# Run error bar fix
rcs = 2
ccs = 1
return_sigma_4 = np.ones(len(pot_slice_4))
while (rcs/ccs > 1.1):
    return_sigma_4, rcs, ccs = determine_sigmas(
            pot_slice_4,
            e3vals[1],
            return_sigma_4)
OG_ERRS = np.append(OG_ERRS, return_sigma_4)
# -----------------------------------------------------------------------
# Plot Raw Data with Error Bars
# -----------------------------------------------------------------------
plt.errorbar(time, pot, yerr=OG_ERRS, capsize=0.5,
        marker='.', markersize=1, alpha=0.2,
        linestyle='', linewidth=0.5,
        zorder=10 ,color='blue', label="data")
plt.legend(loc=1, prop={'size': 6})
plt.xlabel("Time in Seconds")
plt.ylabel("Potential in mV")
plt.savefig("rgpotvtime.pdf")
# =======================================================================
# Print Error Analysis
# =======================================================================
# Equilibrium State One
# -----------------------------------------------------------------------
cs = chi_squared(pot_slice_0, e1vals[1], return_sigma_0)
print(72*'=')
print("Equilibrium State One Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e1[-2])
print("Intercept Standard Deviation:\t",e1[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# -----------------------------------------------------------------------
# Decay State One
# -----------------------------------------------------------------------
cs = chi_squared(pot_slice_1, d1[1], return_sigma_1)
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
# -----------------------------------------------------------------------
# Equilibrium State Two
# -----------------------------------------------------------------------
cs = chi_squared(pot_slice_2, e2vals[1], return_sigma_2)
print("Equilibrium State Two Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e2[-2])
print("Intercept Standard Deviation:\t",e2[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'-')
# -----------------------------------------------------------------------
# Decay State Two
# -----------------------------------------------------------------------
cs = chi_squared(pot_slice_3, d2[1], return_sigma_3)
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
# -----------------------------------------------------------------------
# Equilibrium State Two
# -----------------------------------------------------------------------
cs = chi_squared(pot_slice_4, e3vals[1], return_sigma_4)
print("Equilibrium State One Error Analysis")
print(72*'=')
print("Slope Standard Deviation:\t", e3[-2])
print("Intercept Standard Deviation:\t",e3[-1])
print("Chi Square:\t\t\t",cs[0])
print("Reduced Chi Square:\t\t",cs[1])
print(72*'=')
# =======================================================================
# Calculate G
# =======================================================================
run1=calc_G(216, calc_O(.007832))
run2=calc_G(216, calc_O(.007832-.000855))
print("Run 1 has a calculated G of: ", run1)
gours, gerrp, gerrn = diff_G(run1, .25*10**-11)
print("And a percent error of: ", gours*100)
print("gerrp = ", gerrp*100)
print("gerrn = ", gerrn*100)
gours, gerrp, gerrn = diff_G(run2, .25*10**-11)
print("Run 2 has a calculated G of: ", run2)
print("And a percent error of: ", gours*100)
print("gerrp = ", gerrp*100)
print("gerrn = ", gerrn*100)
