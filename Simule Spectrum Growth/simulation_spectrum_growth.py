# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:14:24 2019

@author: Luciana Martinez

CIBION, Bs As, Argentina

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy.signal as sig

hc = 1239.84193 # Plank's constant times speed of light in eV*nm
k = 0.000086173 # Boltzmann's constant in eV/K


def gaussian(x, Io, xo, wo):

    return Io*np.exp(-2*(x-xo)**2/(wo**2))


def spectrum_NP(Power, wavelength, gamma_spr, londa_spr, starts_notch, ends_notch, laser, To, K):

    last_index_AS = np.where(wavelength >= starts_notch)[0][0]
    wavelength_AS = wavelength[:last_index_AS]

    first_index_S = np.where(wavelength >= ends_notch)[0][0]
    wavelength_S = wavelength[first_index_S:]

    init_params = np.array([gamma_spr, londa_spr], dtype=np.double)

    Temp = To + K*Power

    noise_AS = 0.003*np.random.normal(0, 1, wavelength_AS.shape)
    spectrum_AS = Power*lorentz(wavelength_AS, *init_params)*bose_einstein(wavelength_AS, laser, Temp) # + noise_AS

    noise_S = 0.003*np.random.normal(0, 1, wavelength_S.shape)
    spectrum_S =  Power*lorentz(wavelength_S, *init_params) #+ noise_S

    factor_andor = 1 #30000

    spectrum = np.zeros(len(wavelength))
    spectrum[:last_index_AS] = factor_andor*spectrum_AS
    spectrum[first_index_S:] = factor_andor*spectrum_S

    bkg = background(wavelength)

    #noise = 0.0001*np.random.normal(0, 1, wavelength.shape)
    #spectrum = optical_transmition(wavelength, spectrum) + noise

    spectrum = spectrum #+ bkg

    return Temp, spectrum

def background(wavelength):

    noise = 0.003*np.random.normal(0, 1, wavelength.shape)

    bkg = 300*np.sin((wavelength-wavelength[0])*np.pi/(wavelength[-1]-wavelength[0])) + noise

    return bkg

def smooth(spectrum, window, deg):

    spectrum_smooth = sig.savgol_filter(spectrum, window, deg, mode='mirror')

    return spectrum_smooth

def smooth_matlab(y, window):
    # y: NumPy 1-D array containing the data to be smoothed
    # window: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(y,np.ones(window,dtype=int),'valid')/window   
    r = np.arange(1,window-1,2)
    start = np.cumsum(y[:window-1])[::2]/r
    stop = (np.cumsum(y[:-window:-1])[::2]/r)[::-1]

    return np.concatenate((  start , out0, stop  ))


def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    gamma, x0 = p
    return (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)

def lorentz2(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    I, gamma, x0, C = p
    return I*(gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def bose_einstein(londa, londa_exc, temp):
    # Bose-Einstein distribution
    # temp must be in Kelvin
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    k = 0.000086173 # Boltzmann's constant in eV/K
    kT = k*temp
    inside = hc * (1/londa - 1/londa_exc) / kT
    out = 1 / ( np.exp( inside ) - 1 )

    return out

def NP_function_ratio(r):

    londa_spr = 540 + 5/3*(r-30)
    gamma_spr = 70 + 7/3*(r-30)

    sigma_abs = 8000*(1 + 0.1/30*np.sqrt(r-30))

    return londa_spr, gamma_spr, sigma_abs


def growth_NP(power, wavelength, starts_notch, ends_notch, ro, laser, kappa, To, time, rate_growth):

    rate = rate_growth*power

    spectrum = np.zeros((len(wavelength), len(time)))
    temp = np.zeros(len(time))
    ratio = np.zeros(len(time))
    londa_spr = np.zeros(len(time))
    gamma_spr = np.zeros(len(time))

    for i in range(len(time)):

        ratio[i] = ro + rate*time[i]

        londa_spr[i],  gamma_spr[i], sigma_abs = NP_function_ratio(ratio[i])

        K = sigma_abs / (2*np.pi*kappa*ratio[i]*320**2)

        #Io = power+ 0.05*power*np.sin(1/12*time[i])

        #x=0.5*np.sin(1/12*time[i])
        #po = power+ 0.05*power*np.sin(1/12*time[i])
        #Io = gaussian(x, po, 0, 320)

        Io = power

        temp[i], spectrum[:, i] = spectrum_NP(Io, wavelength, gamma_spr[i], londa_spr[i], starts_notch, ends_notch, laser, To, K)

    return ratio, londa_spr, gamma_spr, temp, spectrum

def fit_lorentz2(p, x, y):
    
    try:
        A = curve_fit(lorentz2, x, y, p0 = p)

    except RuntimeError:
        print("Error - curve_fit failed")
        A = np.zeros(4), np.zeros(4)
    
    return A

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    ssres = ((observed - fitted)**2).sum()
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def fit_spr(spectrum, wavelength, ends_notch):

    desired_range_stokes = np.where((wavelength > ends_notch + 2) & (wavelength <= wavelength[-1] -2))
    wavelength_S = wavelength[desired_range_stokes]
    spectrum_S = spectrum[desired_range_stokes]
    spectrum_S = spectrum_S/max(spectrum_S)

    init_londa = (wavelength[-1] + ends_notch )/2  

    init_params2 = np.array([1, 70, init_londa, 0.0], dtype=np.double)
    best_lorentz, err = fit_lorentz2(init_params2, wavelength_S, spectrum_S)
    
    if best_lorentz[0] != 0:

        lorentz_fitted = lorentz2(wavelength_S, *best_lorentz)
        r2_coef_pearson = calc_r2(spectrum_S, lorentz_fitted)

        full_lorentz_fitted = lorentz2(wavelength, *best_lorentz)
        londa_max_pl = best_lorentz[2]
        width_pl = best_lorentz[1]
        
    else: 
        
        full_lorentz_fitted = np.zeros(len(wavelength))
        londa_max_pl = 0
        width_pl =  0

    return full_lorentz_fitted, londa_max_pl, width_pl

def find_bose_enistein(spectrum, wavelength, starts_notch, lorents_fitted):

    desired_range_antistokes = np.where((wavelength >  wavelength[0] + 2) & (wavelength <= starts_notch -2))
    wavelength_AS = wavelength[desired_range_antistokes]

    spectrum = spectrum/max(spectrum)
    spectrum_AS = spectrum[desired_range_antistokes]

    AS_lorentz = lorents_fitted[desired_range_antistokes]

    AS_bose_einstein = spectrum_AS/AS_lorentz

    return wavelength_AS, AS_bose_einstein


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

def optical_transmition(wavelength, intensity):

	final_intensity = intensity*((0.1/(wavelength[-1]-wavelength[0]))*(wavelength-wavelength[0]) + 0.05)

	return final_intensity

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path


if __name__ == '__main__':
    
    plt.close('all')

    #parent_folder = '//Fileserver/na/Luciana Martinez/Programa_Python/Simule Drift with Confocal Spectrum'
    #parent_folder = 'C:/Users/Alumno/Dropbox/Simule Drift with Confocal Spectrum'
    parent_folder = 'C:/Users/Luciana/Dropbox/Simule Spectrum Growth'
    parent_folder = os.path.normpath(parent_folder)

    meas_pow_bfp = 1 #mW
    factor = 0.47
    Io = factor*meas_pow_bfp

    laser, starts_notch, ends_notch = 532, 525, 540 
    #laser, starts_notch, ends_notch, sigma_abs = 640, 633, 648

    wavelength = np.linspace(500, 600, 1002) #+ 40
    #wavelength = np.linspace(610, 710, 1002) #+ 40

    To = 295

    kappa = 0.8 # water/glass from Setoura et al 2013, in W/m/K
    kappa = kappa*1000*1e-9 # in mW/nm/K
    ratio = 30 # radius in nm

    #K = sigma_abs / (2*np.pi*kappa*a*320**2) # K/mW
    #k= K*(np.pi*0.320**2)/2
   # print('SLOPE T VS IRRADIANCE [K/mW/um2]', k) # K/mW/um2

    time_max = 200 #s
    exposure_time = 2 #s
    liveview_time = exposure_time*1.5
    time = np.arange(0, time_max, liveview_time)

    rate_growth = 30/240 #nm/smW  #30 nm en 4min con 1 mW

    ratio_time, londa_time, gamma_time, Temp_time, spectrums_time = growth_NP(Io, wavelength, starts_notch, ends_notch, ratio, laser, kappa, To, time, rate_growth)
    bkg = background(wavelength)

    max_wavelength = np.zeros(len(time))
    max_wavelength_smooth = np.zeros(len(time))
    max_spectrum = np.zeros(len(time))
    integrate_wavelength_S = np.zeros(len(time))
    integrate_wavelength_AS = np.zeros(len(time))

    desired_range_S = np.where((wavelength > ends_notch) & (wavelength <= wavelength[-1]))
    desired_range_AS = np.where((wavelength >= wavelength[0]) & (wavelength < starts_notch))

    plt.figure()
    for i in range(len(time)):
        plt.plot(wavelength, spectrums_time[:,i])
        plt.ylabel('Intensity')
        plt.xlabel('Wavelength')
        #plt.legend()
        desired_wavelength =  wavelength[desired_range_S]
        max_wavelength[i] = desired_wavelength[np.argmax(spectrums_time[:,i][desired_range_S])]

        smooth_spectrum = smooth(spectrums_time[:,i], window = 41, deg = 0)
        smooth_bkg =  smooth(bkg, window = 41, deg = 0)
        smooth_spectrum = smooth_spectrum - smooth_bkg 

        max_wavelength_smooth[i] = desired_wavelength[np.argmax(smooth_spectrum[desired_range_S])]

        max_spectrum[i] = np.max(spectrums_time[:,i][desired_range_S])

        integrate_wavelength_S[i] = np.mean(spectrums_time[:,i][desired_range_S])
        integrate_wavelength_AS[i] = np.mean(spectrums_time[:,i][desired_range_AS])

    plt.show()


    plt.figure()
    plt.plot(time, integrate_wavelength_S, 'o', label = 'integrate stokes')
    plt.plot(time, max_spectrum, '-o', label = 'maximum intensity stokes')
    plt.plot(time, integrate_wavelength_AS, '-o', label = 'integrate Anti-Stokes')
    plt.xlabel('Time (s)')
    plt.ylabel('Integrate wavelength')
    plt.legend()
    plt.show()

    londa_array = np.zeros(len(time))
    width_array = np.zeros(len(time))

   
    window, deg = 41, 0
    for i in range(len(time)):
        #spectrums_time[:,i] = smooth(spectrums_time[:,i]- bkg, window = window, deg = deg)
        spectrums_time[:,i] = smooth_matlab(spectrums_time[:,i]-bkg, window = 41)

    plt.figure()
    for i in range(len(time)):
        full_lorentz_fitted, fit_londa_spr, fit_width_spr = fit_spr(spectrums_time[:,i], wavelength, ends_notch)
        londa_array[i] =  fit_londa_spr
        width_array[i] =  fit_width_spr
        wave_AS, bose_einstein = find_bose_enistein(spectrums_time[:,i], wavelength, starts_notch, full_lorentz_fitted)
        plt.plot(wavelength, spectrums_time[:,i]/max(spectrums_time[:,i]))
        plt.plot(wavelength, full_lorentz_fitted,'k--')
        
    plt.ylabel('Intensity (a.u.)')
    plt.xlabel('Wavelength (nm)')
    plt.ylim([0,1.05])
    plt.show()


    Temp_BE_1 = np.zeros(len(time))
    err_Temp_BE = np.zeros(len(time))


    plt.figure()
    for i in range(len(time)):
        full_lorentz_fitted, fit_londa_spr, fit_width_spr = fit_spr(spectrums_time[:,i], wavelength, ends_notch)
        wave_AS, bose_einstein = find_bose_enistein(spectrums_time[:,i], wavelength, starts_notch, full_lorentz_fitted)
        inside = np.log(1/bose_einstein+1)
       # plt.plot(wave_AS, bose_einstein, 'm--')
        m, c, err_m, err_c, r_squared, x_fitted, y_fitted = fit_linear(1/wave_AS, inside, weights = [])

        if r_squared > 0.8:
            Temp_BE_1[i] = hc/(m*k)
            err_Temp_BE[i] = Temp_BE_1[i]*(err_m/m)

        plt.plot(1/wave_AS, inside, '-')
        plt.plot(x_fitted, y_fitted, 'k--')

    plt.ylabel('inside bose-einstein')
    plt.xlabel('1/Wavelength (nm)')
    #aux_folder = manage_save_directory(save_folder,'spr')
    #figure_name = os.path.join(aux_folder, 'specs_fitted_lorentz_%s.png' % time[i])
    #plt.savefig(figure_name)
    plt.show()
    

    Temp_BE = Temp_BE_1[np.where(Temp_BE_1>0)[0]]
    err_Temp_BE = err_Temp_BE[np.where(Temp_BE_1>0)[0]]
    time_BE = time[np.where(Temp_BE_1>0)[0]]
    londa = londa_array[np.where(Temp_BE_1>0.5)[0]]

    plt.figure()
    plt.plot(time, Temp_time, '--', label = 'signal')
    plt.errorbar(time_BE, Temp_BE, yerr = err_Temp_BE, fmt = '*', label = 'fitting bose-einstein')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.ylim(Temp_time[-1] - Temp_time[-1]*0.5, Temp_time[0] + Temp_time[0]*0.5)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, londa_time, '--', label = 'signal')
    plt.plot(time, londa_array, '-*', label = 'post fitting')
    plt.plot(time, max_wavelength, 'o', label = 'LIVE')
    plt.plot(time, max_wavelength_smooth, 'o', label = 'LIVE SMOOTH')
    plt.xlabel('Time (s)')
    plt.ylabel('Max wavelength SPR (nm)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, gamma_time, '--', label = 'signal')
    plt.plot(time, width_array, '-*', label = 'post fitting')
    plt.xlabel('Time (s)')
    plt.ylabel('Width SPR (nm)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(londa_time, Temp_time, '--', label = 'signal')
    plt.errorbar(londa , Temp_BE, yerr = err_Temp_BE, fmt = '-*', label = 'fitting')
    plt.plot(londa , smooth(Temp_BE, 5, 3), '-*')
    plt.xlabel('Max wavelength SPR (nm)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.show()
