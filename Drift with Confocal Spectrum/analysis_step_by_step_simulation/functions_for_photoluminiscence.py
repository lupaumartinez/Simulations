# -*- coding: utf-8 -*-
"""
Auxiliary functions of "witec_data_photoluminiscence.py"

Mariano Barella

16 oct 2018

Imperial College, London, UK
"""

import numpy as np
from scipy.optimize import curve_fit
import os

## Uncomment these lines to ignore any warning message
import warnings
warnings.filterwarnings("ignore")

## DETERMINACION DE PARAMETROS
h = 4.135667516e-15 # in eV*s
c = 299792458 # in m/s

def fit_linear(x, y, weights, intercept=True):
    # fit y=f(x) as a linear function
    # intercept True: y = m*x+c (non-forcing zero)
    # intercept False: y = m*x (forcing zero)
    # weights: used for weighted fitting. If empty, non weighted

    x = np.array(x)
    y = np.array(y)
    weights = np.array(weights)
    N = len(x)
    x_original = x
    y_original = y
    # calculation of weights for weaighted least squares
    if weights.size == 0:
        print('Ordinary Least-Squares')
        x = x
        y = y
        ones = np.ones(N)
    else:
        print('Weighted Least-Squares')
        x = x * np.sqrt(weights)
        y = y * np.sqrt(weights)
        ones = np.ones(N) * np.sqrt(weights)
    # set for intercept true or false
    if intercept:
        indep = ones
    else:
        indep = np.zeros(N)

    # do the fitting
    if N > 1:
        print('More than 1 point is being fitted.')
        A = np.vstack([x, indep]).T
        p, residuals, _, _ = np.linalg.lstsq(A, y)
#        print('Irradiance', x)
#        print('Temp. increase', y)
#        print('Slope', p[0], 'Offset', p[1])
        x_fitted = np.array(x_original)
        y_fitted = np.polyval(p, x_fitted)
        # calculation of goodess of the fit
        y_mean = np.mean(y)
        SST = sum([(aux2 - y_mean)**2 for aux2 in y_original])
        SSRes = sum([(aux3 - aux2)**2 for aux3, aux2 in zip(y_original, y_fitted)])
        r_squared = 1 - SSRes/SST
        # calculation of parameters and errors
        m = p[0]
        sigma = np.sqrt(np.sum((y_original - p[0]*x_original - p[1])**2) / (N - 2))
        aux_err_lstsq = (N*np.sum(x_original**2) - np.sum(x_original)**2)
        err_m = sigma*np.sqrt(N / aux_err_lstsq)
        if intercept:
            c = p[1]
            err_c = sigma*np.sqrt(np.sum(x_original**2) / aux_err_lstsq)
        else:
            c = 0
            err_c = 0
    # elif N == 0:
    #     print('No points to fit')
    #     m = 0
    #     err_m = 0
    #     c = 0
    #     err_c = 0
    #     r_squared = 0
    #     x_fitted = x
    #     y_fitted = y
    else:
        print('One single point. No fitting performed.')
        m = 0
        err_m = 0
        c = 0
        err_c = 0
        r_squared = 0
        x_fitted = x
        y_fitted = y

    return m, c, err_m, err_c, r_squared, x_fitted, y_fitted

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def grc_filter(array,threshold,msg):
    # Filter spikes originated by Galactic Cosmis Rays or similar in the CCD signal (i.e. the array).
    # threshold should be above 1, for instance 1.1 to filter a spike 10% higher than its neighbours
    # msg would print the number of peaks filtered
    counter = 0
    flag = False
    for i in range(len(array)-1):
        aux = array[i]
        if ((aux > threshold*array[i-1]) and (aux > threshold*array[i+1])):
            counter += 1
            array[i] = (array[i-1] + array[i+1])/2
#            array[i] = array[i-1]
            flag = True
    if flag:
        if msg:
            print('Warning: %d GCR peak(s) detected! Data was modified.' % counter)
    return array

def classification(value, totalbins, rango):
    # Bin the data. Classify a value into a bin.
    # totalbins = number of bins to divide rango (range)
    bin_max = totalbins - 1
    numbin = 0
    inf = rango[0]
    sup = rango[1]
    if value > sup:
        print('Value higher than max')
        return bin_max
    if value < inf:
        print('Value lower than min')
        return 0
    step = (sup - inf)/totalbins
    # tiene longitud totalbins + 1
    # pero en total son totalbins "cajitas" (bines)
    binned_range = np.arange(inf, sup + step, step)
    while numbin < bin_max:
        if (value >= binned_range[numbin] and value < binned_range[numbin+1]):
            break
        numbin += 1
        if numbin > bin_max:
            break
    return numbin

def lambda_to_energy(londa):
    # Energy in eV and wavelength in nm
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    energy = hc/londa
    return energy

def energy_to_lambda(energy):
    # Energy in eV and wavelength in nm
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    londa = hc/energy
    return londa

def gaussian2(x, *p):
    # Gaussian fitting function
    # I = amplitude
    # x0 = center
    # c = offset
    I, w0, x0, c = p
    return I * np.exp(-2 * (x - x0)**2 / w0**2) + c

def gaussian(x, *p):
    # Gaussian fitting function
    # I = amplitude
    # x0 = center
    # c = offset
    I, w0, x0 = p
    return I * np.exp(-2 * (x - x0)**2 / w0**2)

def lorentz2(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = 3.141592653589793
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def lorentz(x, *p):
    # Lorentz fitting function without an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = 3.141592653589793
    I, gamma, x0 = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)
    
def quotient(E, *p):
    # Antistokes quotient fitting function
    # A is laser power 2 divided by laser power 1
    # E is the energy array and A is the amplitude 
    k = 0.000086173 # Boltzmann's constant in eV/K
    T0 = 298.15 # room temperature in K
    El = 2.33 # excitation wavelength in eV
    T1, A = p
    T2 = A * (T1 - T0) + T0
    a1 = (E-El) / ( 2*k*T1 )
    a2 = (E-El) / ( 2*k*T2 )
    return A * np.exp( -(a2 - a1) ) * np.sinh(a1) / np.sinh(a2)

def bose_einstein(londa, temp):
    # Bose-Einstein distribution
    # temp must be in Kelvin
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    k = 0.000086173 # Boltzmann's constant in eV/K
    londa_exc = 532 # in nm
    kT = k*temp
    inside = hc * (1/londa - 1/londa_exc) / kT
    out = 1 / ( np.exp( inside ) - 1 )
    return out

def nanothermometry(londa, lorentz_spr, temp):
    # Auxiliary function to apply Carattino et al strategy
    if not len(londa) == len(lorentz_spr):
        print('Error! londa must have the same length as lorentz_spr.')
        return -1
    out = lorentz_spr*bose_einstein(londa,temp)
    return out

def factor(intensity_theo, factor):
    # Factor fitting function
    out = intensity_theo*factor
    return out

def sum_of_signals(londa, raman, gamma, x0, C, amplitude1, amplitude2):
    # Functions that returns the sum of two signals
    # PL and Raman
    # gamma = FWHM
    # x0 = center
    pi = 3.141592653589793
    lorentz = (1/pi) * amplitude1 * (gamma/2)**2 / ((londa - x0)**2 + (gamma/2)**2) + C
    out = lorentz + amplitude2*raman
    return out

def fit_factor(p, x, y):
    return curve_fit(factor, x, y, p0 = p)

def fit_gaussian(p, x, y):
    return curve_fit(gaussian, x, y, p0 = p)

def fit_gaussian2(p, x, y):
    return curve_fit(gaussian2, x, y, p0 = p)

def fit_lorentz(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)

def fit_lorentz2(p, x, y):
    return curve_fit(lorentz2, x, y, p0 = p)
    
def fit_quotient(p, x, A, err_A, y):
    delta = 0.001*A
#    delta = 0.01
#    delta = err_A
    return curve_fit(quotient, x, y, p0 = p, bounds=([0,A-delta], [5000,A+delta]))

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def calc_chi_squared(observed, expected):
    # Chi-squared (pearson) coefficient calculation
    aux = (observed - expected)**2/expected
    ans = aux.sum()
    return ans

