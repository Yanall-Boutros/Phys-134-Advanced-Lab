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
import csv
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
nlvgp = [3300, 0.025, 193072*np.pi*2, 361152, 0.09138]
lvgp = [100, 0.015, 13087*2*np.pi, 15861, 3.215]

def volt(xdata, R=1, L=1, resfrq=1, Gamma=1, V0=1):
    num = V0 * xdata * (R/L)
    den = ((resfrq**2 - xdata ** 2)**2 + (xdata*Gamma)**2)**0.5
    return num/den

nonlin_phi_gp = [193072*np.pi*2, 361152]
lin_phi_gp = [13086*np.pi*2, 15861]

def phi(xdata, res_freq=81640, Gamma=100000):
    return np.arctan(-1*(xdata**2-res_freq**2)/(xdata*Gamma))

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

# =======================================================================
# Read in files from directory
# =======================================================================
with open('nonLinPhaseAmp.csv') as csvfile:
    data = list(csv.reader(csvfile))
with open('phaseAmp.csv') as csvfile:
    lindata = list(csv.reader(csvfile))
# =======================================================================
# Data Initalization
# =======================================================================
phase = []
res_volt = []
inp_volt = []
phase_err = []
res_err = []
inp_err = []
infrq = []
inferr = []
outfreq = []
outfreqerr = []
for i in range(1, len(data)):
    phase.append(float(data[i][0]))
    res_volt.append(float(data[i][1]))
    inp_volt.append(float(data[i][2]))
    phase_err.append(float(data[i][3]))
    res_err.append(float(data[i][4]))
    inp_err.append(float(data[i][5]))
    infrq.append(float(data[i][6]))
    inferr.append(float(data[i][7]))
    outfreq.append(float(data[i][8]))
    outfreqerr.append(float(data[i][9]))
phase = np.array(phase)
res_volt = np.array(res_volt)
inp_volt = np.array(inp_volt)
phase_err = np.array(phase_err)
res_err = np.array(res_err)
inp_err = np.array(inp_err)
infrq = np.array(infrq)
inferr = np.array(inferr)
outfreq = np.array(outfreq)
outfreqerr = np.array(outfreqerr)
# -----------------------------------------------------------------------
# lin data
# -----------------------------------------------------------------------
lin_phase = []
lin_res_volt = []
lin_inp_volt = []
lin_phase_err = []
lin_res_err = []
lin_inp_err = []
lin_infrq = []
lin_inferr = []
lin_outfreq = []
lin_outfreqerr = []
for i in range(1, len(lindata)):
    lin_phase.append(float(lindata[i][0].lstrip()))
    lin_res_volt.append(float(lindata[i][1].lstrip()))
    lin_inp_volt.append(float(lindata[i][2].lstrip()))
    lin_phase_err.append(float(lindata[i][3].lstrip()))
    lin_res_err.append(float(lindata[i][4].lstrip()))
    lin_inp_err.append(float(lindata[i][5].lstrip()))
    lin_infrq.append(float(lindata[i][6].lstrip()))
    lin_inferr.append(float(lindata[i][7].lstrip()))
lin_phase = np.array(lin_phase)
lin_res_volt = np.array(lin_res_volt)
lin_inp_volt = np.array(lin_inp_volt)
lin_phase_err = np.array(lin_phase_err)
lin_res_err = np.array(lin_res_err)
lin_inp_err = np.array(lin_inp_err)
lin_infrq = np.array(lin_infrq)
lin_inferr = np.array(lin_inferr)
# ------------------------------------------------------------------------
# Fitting
# ------------------------------------------------------------------------
# Nonlinear phase and freq
nlpf = LeastSquaresFit(2*np.pi*infrq, np.deg2rad(phase),
        100*np.ones(len(infrq)), phi,
        nonlin_phi_gp)
lpf = LeastSquaresFit(2*np.pi*lin_infrq,
        np.deg2rad(lin_phase),
        100*np.ones(len(lin_infrq)), phi, lin_phi_gp)
# ------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------
# Nonlinear phase vs freq
plt.plot(nlpf[0]/(2*np.pi), np.rad2deg(nlpf[1]), linestyle="--")
plt.errorbar(infrq, phase, 
             xerr=inferr, yerr=phase_err,
             linestyle="none", marker=".", label="Raw Data")
plt.title("Nonlinear Phase versus Input Frequency")
plt.xlabel("Frequency $Hz$")
plt.ylabel("Phase $\deg$")
plt.savefig("nonlinPhasevsFreq.pdf")

# Linear Phase vs Freq
plt.figure()
plt.plot(lpf[0]/(2*np.pi), np.rad2deg(lpf[1]), linestyle="--")
plt.errorbar(lin_infrq, lin_phase, 
             xerr=lin_inferr, yerr=lin_phase_err,
             linestyle="none", marker=".", label="Raw Data")
plt.title("Linear Phase versus Input Frequency")
plt.xlabel("Frequency $Hz$")
plt.ylabel("Phase $\deg$")
plt.savefig("linPhasevsFreq.pdf")
plt.figure()
# Nonlinear volt vs freq
plt.errorbar(infrq, res_volt, xerr=inferr, yerr=res_err,
        linestyle="none", marker=".", label="Raw Data")
plt.ylim(0, 0.04)

plt.title("Nonlinear Response Voltage vs Input Frequency")
plt.xlabel("Frequency $Hz$")
plt.ylabel("Voltage (Amps)")
plt.savefig("VoltvsFreq.pdf")
plt.figure()
#linear volt vs freq
plt.errorbar(lin_infrq, lin_res_volt, xerr=lin_inferr, yerr=lin_res_err,
        linestyle="none", marker=".", label="Raw Data")
plt.title("Linear Response Voltage vs Input Frequency")
plt.xlabel("Freqency $Hz$")
plt.ylabel("Voltage (Amps)")
plt.savefig("LinVoltvsFreq.pdf")
