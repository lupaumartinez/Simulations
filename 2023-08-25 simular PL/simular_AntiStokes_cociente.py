# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:14:47 2023

@author: Luciana
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:11:52 2023

@author: lupau
"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

## DETERMINACION DE PARAMETROS
h = 4.135667516e-15 # in eV*s
c = 299792458 # in m/s

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

def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = 3.141592653589793
    x0, gamma = p
    return (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2)

def temp_with_slope(Tzero, slope, Irrad):
    T = slope*Irrad + Tzero
    return T

def bose(energy, El, Temp):
    k = 0.000086173 # Boltzmann's constant in eV/K
    aux = (energy - El) / (k*Temp)
    y = 1/( np.exp(aux) - 1 )
    return y

def fit_quotient_for_beta(p, x, laser, Tzero, Irrad_n, Irrad_m, y, error_y):
    quotient_lambda_func = lambda energy, beta : quotient_with_slope(energy, laser, Tzero, beta, Irrad_n, Irrad_m)
    return curve_fit(quotient_lambda_func, x, y, p0 = p, bounds=(0, 5000), sigma=error_y, absolute_sigma=True)

def quotient_with_slope(energy, laser, Tzero, slope, Irrad_n, Irrad_m):
    # Antistokes quotient fitting function
    # E is the energy array
    k = 0.000086173 # Boltzmann's constant in eV/K
    El = lambda_to_energy(laser) # excitation wavelength in eV
    Tn = temp_with_slope(Tzero, slope, Irrad_n)
    Tm = temp_with_slope(Tzero, slope, Irrad_m)
    
  #  print('Irrad_n', Irrad_n, 'Irrad_m', Irrad_m)
    
    aux_n = (energy - El) / ( k*Tn )
    aux_m = (energy - El) / ( k*Tm )
    # two equivalent expressions can be used for the quotient
    # reduced form:
#    quotient = A * np.exp( -(aux_n - aux_m) ) * np.sinh(aux_m) / np.sinh(aux_n)
    # explicit form:
    bose_m = 1/( np.exp(aux_m) - 1 )
    bose_n = 1/( np.exp(aux_n) - 1 )
    quotient = (Irrad_n * bose_n) / ( Irrad_m * bose_m )
    return quotient

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

def add_white_noise(signal, mean, std):
    noise = np.random.normal(mean, std, signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

def Q_AS(long, laser, size_notch, QY, betha, Tmax, To, nbins, omega, m_ref, noise):
    
   # print('laser', laser, 'betha', betha)
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    long_AS = long[desired_AS]
   # desired_AS = np.where(long_AS > long_AS[0] + 7)
  #  long_AS = long_AS[desired_AS]
    
    energyAS = lambda_to_energy(long_AS)
    El = lambda_to_energy(laser) # excitation wavelength in eV
    
    e = energyAS - El
    desired = np.where((e > 0.05) & (e < 0.09))
    energyAS = energyAS[desired]
    energy_shift = e[desired]
 
    r = np.linspace(0, np.sqrt(2*0.400**2), nbins)
    g = np.exp(-2*r**2/omega**2)
    g = np.flip(g)
    
    Imax = round((Tmax-To)/betha,4)
    I_array = g*Imax
    
    T_array = betha*I_array + To
    
    AS = np.zeros((nbins, len(energyAS)))
    errorAS = np.zeros((nbins, len(energyAS)))
    
    bkg =  np.ones(len(energyAS)) #np.random.normal(0,0.05,len(energyAS))
    
    plt.figure()
    #plt.xlabel('Wavelength (nm)')
    plt.xlabel('energy shift (eV)')
    plt.ylabel('I*BE')
  #  plt.ylim(1, 2.5)
    
    totalbins = nbins - 1
    
 #   print( 'total bins',   totalbins)
    
    ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    error_ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    
    for i in range(nbins):
        
        T = T_array[i]
        I = I_array[i]
        
        BE = bose(energyAS, El, T)
        
        signal = QY*I*BE + bkg
        signal = add_white_noise(signal, 0, noise)
        
        plt.plot(energy_shift, signal, '.-', label = i)
        
        smooth_signal = savgol_filter(signal, 51, 3, mode = 'interp')
        
        AS[i, :] = smooth_signal
        
        error_signal = np.sqrt((smooth_signal-signal)**2)/2
        
        errorAS[i, :] = error_signal
        
        plt.errorbar(energy_shift, smooth_signal, yerr = error_signal, linestyle = '--', color = 'k')
        
 #   bkg = AS[0, :]
        
    for k in range(totalbins):
        
        AS_ref = AS[k+1, :] - bkg
        
        errorAS_ref = errorAS[k+1, :]
    
       # plt.figure()
       # plt.title('ref: %s'%(k))
       # plt.xlabel('Energy (eV)')
       # plt.ylabel('Q')
            
        for j in range(totalbins):
            
            AS_j = AS[j+1, :]- bkg
            
            errorAS_j = errorAS[j+1, :]
            
            Q =  AS_j/AS_ref
            
            errorQ = ( errorAS_j/AS_j + errorAS_ref/AS_ref )*Q
            
            ydata_all[j,k,:] = Q
            
            error_ydata_all[j,k,:] = errorQ
            
          #  plt.plot(energyAS, Q, label = '%s,ref: %s'%(j,k))
        
      #  plt.legend()
        
    #plt.close('all')
   
    Irrad_array = I_array[1:]
    err_Irrad_array = 0.01*Irrad_array
            
    R2_matrix = np.zeros([totalbins, totalbins])
    chi_squared_matrix = np.zeros([totalbins, totalbins])
    p_value_matrix = np.zeros([totalbins, totalbins])
    beta_matrix = np.zeros([totalbins, totalbins])
    err_beta_matrix = np.zeros([totalbins, totalbins])
    Tzero_matrix = np.zeros([totalbins, totalbins])
    err_Tzero_matrix = np.zeros([totalbins, totalbins])
    T_matrix = np.zeros([totalbins, totalbins])
    err_T_matrix = np.zeros([totalbins, totalbins])
        
    skip_neighbours = 0
    x = energyAS 
    rango = list(range(0, totalbins))
    norm = plt.Normalize()
    colormap = plt.cm.plasma(norm(rango))
    
    Elaser = lambda_to_energy(laser)
    #last_bin = list(range(0, totalbins))
    
    bethas = []
    
    plot = False
           
    for ref in range(totalbins-1):
        print('\nFitting quotient Q using bin %d as reference...' % ref)
        
        list_of_bins_to_fit = list(range(ref+1+skip_neighbours, totalbins))
        
        if plot:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 0.15)
        
        for i in list_of_bins_to_fit:
            
            if m_ref:
            
                # m = ref
                Irrad_n = Irrad_array[i]
                Irrad_m = Irrad_array[ref]
                err_Irrad_n = err_Irrad_array[i]
                err_Irrad_m = err_Irrad_array[ref]
                y = ydata_all[i, ref, :]
                error_y = error_ydata_all[i, ref, :]
            
            else:
                # n = ref
                Irrad_m = Irrad_array[i]
                Irrad_n = Irrad_array[ref]
                err_Irrad_m = err_Irrad_array[i]
                err_Irrad_n = err_Irrad_array[ref]
                y = ydata_all[ref,i, :]
                error_y = error_ydata_all[ref,i, :]
            
           # print('irradiance', Irrad_n, Irrad_m, y)
            
            Tzero = To
            init_params = betha 
            best_as, err = fit_quotient_for_beta(init_params, x, laser, Tzero, Irrad_n, Irrad_m, y, error_y)
                # retrieve best fitted parameters
            b = best_as[0]
            Tref = temp_with_slope(Tzero, b, Irrad_m)
            Tbin = temp_with_slope(Tzero, b, Irrad_n)
            
                # calculate the errors and goodes of the fit
            err_beta = np.sqrt(err[0,0])
            err_Tzero = 0.5
            err_Tref = np.sqrt(err_Tzero**2 + (Irrad_m*err_beta)**2 + (err_Irrad_m*b)**2)
            err_Tbin = np.sqrt(err_Tzero**2 + (Irrad_n*err_beta)**2 + (err_Irrad_n*b)**2)
            yfit = quotient_with_slope(x, laser, Tzero, b, Irrad_n, Irrad_m)
            r2_coef_pearson = calc_r2(y, yfit)
            chi_squared_pearson = calc_chi_squared(y, yfit)
                # Plotting
            x_long = x
            yfit_to_plot = quotient_with_slope(x_long, laser, Tzero, b, Irrad_n, Irrad_m) 
            
            if plot:
                color_iter = colormap[i]
                
                ax.errorbar(x-Elaser, y, yerr = error_y, fmt = '.', linewidth = 2.0, color = color_iter, label='$Q_{%d/%d}$' % (i, ref))
                
                ax.plot(x_long-Elaser, yfit_to_plot, '--k', alpha = 0.8)
          
                # Asign matrix elements
            beta_matrix[ref,i] = b
            err_beta_matrix[ref,i] = err_beta
            Tzero_matrix[ref,i] = Tzero
            err_Tzero_matrix[ref,i] = err_Tzero
            T_matrix[ref,i] = Tref
            T_matrix[i,ref] = Tbin
            err_T_matrix[ref,i] = err_Tref
            err_T_matrix[i,ref] = err_Tbin
            R2_matrix[ref,i] = r2_coef_pearson
            R2_matrix[i,ref] = r2_coef_pearson
            
            if r2_coef_pearson > 0.8:
                bethas.append(b)
            
            chi_squared_matrix[ref,i] = chi_squared_pearson
            chi_squared_matrix[i,ref] = chi_squared_pearson
        #    p_value_matrix[ref,i] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    p_value_matrix[i,ref] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    print('---------- Bin', i, \
          #            '\nR-sq: %.3f' % r2_coef_pearson, \
          #            '\nTzero: %.1f' % Tzero, 'err Tzero: %.1f' % err_Tzero, \
            #          '\nbeta: %.1f' % beta, 'error beta: %.2f' % err_beta, \
               #       '\nT_ref: %.1f' % Tref, 'error T_ref: %.1f' % err_Tref, \
               #       '\nT_bin: %.1f' % Tbin, 'error T_bin: %.1f' % err_Tbin)
               
          #  print('---------- Bin', i, \
                #      '\nR-sq: %.3f' % r2_coef_pearson, \
                #      '\nTzero: %.1f' % Tzero,  \
                 #     '\nbeta: %.1f' % b,  \
                 #     '\nT_ref: %.1f' % Tref, \
                   #   '\nT_bin: %.1f' % Tbin)
        if plot:        
            fig.legend()
            
    return Irrad_array, beta_matrix, bethas


def Q_AS_irradiance(long, laser, size_notch, QY, betha, Imax, To, nbins, omega, m_ref, noise):
    
   # print('laser', laser, 'betha', betha)
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    long_AS = long[desired_AS]
   # desired_AS = np.where(long_AS > long_AS[0] + 7)
  #  long_AS = long_AS[desired_AS]
    
    energyAS = lambda_to_energy(long_AS)
    El = lambda_to_energy(laser) # excitation wavelength in eV
    
    e = energyAS - El
    desired = np.where((e > 0.05) & (e < 0.09))
    energyAS = energyAS[desired]
    energy_shift = e[desired]
    
  #  e = energyAS - El
    
 #   desired = np.where(e < min(e) + 0.03)
 #   energyAS = energyAS[desired]
    
    r = np.linspace(0, np.sqrt(2*0.400**2), nbins)
    g = np.exp(-2*r**2/omega**2)
    g = np.flip(g)

    I_array = Imax*g

    T_array = betha*I_array + To
    
    AS = np.zeros((len(T_array), len(energyAS)))
    errorAS = np.zeros((len(T_array), len(energyAS)))
    
    bkg =  np.ones(len(energyAS)) #np.random.normal(0,0.05,len(energyAS))
    
    plt.figure()
    #plt.xlabel('Wavelength (nm)')
    plt.xlabel('energy (eV)')
    plt.ylabel('I*BE')
  #  plt.ylim(1, 2.5)
    
    totalbins = len(T_array) - 1
    
 #   print( 'total bins',   totalbins)
    
    ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    
    error_ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    
    for i in range(len(I_array)):
        
        T = T_array[i]
        
        I = I_array[i]
        
        BE = bose(energyAS, El, T)
        
        signal = QY*I*BE + bkg
        
        signal = add_white_noise(signal, 0, noise)
        
        plt.plot(energy_shift, signal, '.', label = i)
        
        
        smooth_signal = savgol_filter(signal, 51, 1, mode = 'interp')
        
        AS[i, :] = smooth_signal
        
        error_signal = np.sqrt((smooth_signal-signal)**2)/2
        
        errorAS[i, :] = error_signal

        #signal = savgol_filter(signal, 51, 1, mode = 'interp')
        
        
        plt.plot(energy_shift, AS[i, :], 'k--')
        
    for k in range(totalbins):
        
        AS_ref = AS[k+1, :] - bkg
        errorAS_ref = errorAS[k+1, :]
    
       # plt.figure()
       # plt.title('ref: %s'%(k))
       # plt.xlabel('Energy (eV)')
       # plt.ylabel('Q')
            
        for j in range(totalbins):
            
            AS_j = AS[j+1, :]- bkg
            errorAS_j = errorAS[j+1, :]
            
            Q =  AS_j/AS_ref
            
            ydata_all[j,k,:] = Q
            
            errorQ = ( errorAS_j/AS_j + errorAS_ref/AS_ref )*Q
            
            error_ydata_all[j,k,:] = errorQ
            
          #  plt.plot(energyAS, Q, label = '%s,ref: %s'%(j,k))
        
      #  plt.legend()
        
    #plt.close('all')
   
    Irrad_array = I_array[1:]
    err_Irrad_array = 0.01*Irrad_array
            
    R2_matrix = np.zeros([totalbins, totalbins])
    chi_squared_matrix = np.zeros([totalbins, totalbins])
    p_value_matrix = np.zeros([totalbins, totalbins])
    beta_matrix = np.zeros([totalbins, totalbins])
    err_beta_matrix = np.zeros([totalbins, totalbins])
    Tzero_matrix = np.zeros([totalbins, totalbins])
    err_Tzero_matrix = np.zeros([totalbins, totalbins])
    T_matrix = np.zeros([totalbins, totalbins])
    err_T_matrix = np.zeros([totalbins, totalbins])
        
    skip_neighbours = 0
    x = energyAS 
    rango = list(range(0, totalbins))
    norm = plt.Normalize()
    colormap = plt.cm.plasma(norm(rango))
    
    Elaser = lambda_to_energy(laser)
    #last_bin = list(range(0, totalbins))
    
    bethas = []
    
    plot = False
           
    for ref in range(totalbins-1):
        print('\nFitting quotient Q using bin %d as reference...' % ref)
        
        #plt.figure(num=1,clear=True)
        list_of_bins_to_fit = list(range(ref+1+skip_neighbours, totalbins))
       # list_of_bins_to_fit = list(range(totalbins-1, ref+1+skip_neighbours-1, -1))
        
       # ax.set_ylim(0, 1) #50)
       
        if plot:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 0.15)
            
        
        for i in list_of_bins_to_fit:
            
            if m_ref:
            
                # m = ref
                Irrad_n = Irrad_array[i]
                Irrad_m = Irrad_array[ref]
                err_Irrad_n = err_Irrad_array[i]
                err_Irrad_m = err_Irrad_array[ref]
                y = ydata_all[i, ref, :]
                error_y = error_ydata_all[i,ref, :]
            
            else:
                # n = ref
                Irrad_m = Irrad_array[i]
                Irrad_n = Irrad_array[ref]
                err_Irrad_m = err_Irrad_array[i]
                err_Irrad_n = err_Irrad_array[ref]
                y = ydata_all[ref,i, :]
                error_y = error_ydata_all[ref,i, :]
            
           # print('irradiance', Irrad_n, Irrad_m, y)
            
            Tzero = To
            init_params = betha 
            best_as, err = fit_quotient_for_beta(init_params, x, laser, Tzero, Irrad_n, Irrad_m, y, error_y)
                # retrieve best fitted parameters
            b = best_as[0]
            Tref = temp_with_slope(Tzero, b, Irrad_m)
            Tbin = temp_with_slope(Tzero, b, Irrad_n)
            
                # calculate the errors and goodes of the fit
            err_beta = np.sqrt(err[0,0])
            err_Tzero = 0.5
            err_Tref = np.sqrt(err_Tzero**2 + (Irrad_m*err_beta)**2 + (err_Irrad_m*b)**2)
            err_Tbin = np.sqrt(err_Tzero**2 + (Irrad_n*err_beta)**2 + (err_Irrad_n*b)**2)
            yfit = quotient_with_slope(x, laser, Tzero, b, Irrad_n, Irrad_m)
            r2_coef_pearson = calc_r2(y, yfit)
            chi_squared_pearson = calc_chi_squared(y, yfit)
                # Plotting
        #    x_long = x
        #    yfit_to_plot = quotient_with_slope(x_long, laser, Tzero, b, Irrad_n, Irrad_m) 
            
            if plot:
            
                color_iter = colormap[i]
                ax.plot(x-Elaser, y, '-', linewidth = 2.0, color = color_iter, label='$Q_{%d/%d}$' % (i, ref))
                ax.plot(x-Elaser, yfit, '--k', alpha = 0.8)
          
                # Asign matrix elements
            beta_matrix[ref,i] = b
            err_beta_matrix[ref,i] = err_beta
            Tzero_matrix[ref,i] = Tzero
            err_Tzero_matrix[ref,i] = err_Tzero
            T_matrix[ref,i] = Tref
            T_matrix[i,ref] = Tbin
            err_T_matrix[ref,i] = err_Tref
            err_T_matrix[i,ref] = err_Tbin
            R2_matrix[ref,i] = r2_coef_pearson
            R2_matrix[i,ref] = r2_coef_pearson
            
            if r2_coef_pearson > 0.8:
                bethas.append(b)
            
            chi_squared_matrix[ref,i] = chi_squared_pearson
            chi_squared_matrix[i,ref] = chi_squared_pearson
        #    p_value_matrix[ref,i] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    p_value_matrix[i,ref] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    print('---------- Bin', i, \
          #            '\nR-sq: %.3f' % r2_coef_pearson, \
          #            '\nTzero: %.1f' % Tzero, 'err Tzero: %.1f' % err_Tzero, \
            #          '\nbeta: %.1f' % beta, 'error beta: %.2f' % err_beta, \
               #       '\nT_ref: %.1f' % Tref, 'error T_ref: %.1f' % err_Tref, \
               #       '\nT_bin: %.1f' % Tbin, 'error T_bin: %.1f' % err_Tbin)
               
          #  print('---------- Bin', i, \
                #      '\nR-sq: %.3f' % r2_coef_pearson, \
                #      '\nTzero: %.1f' % Tzero,  \
                 #     '\nbeta: %.1f' % b,  \
                 #     '\nT_ref: %.1f' % Tref, \
                   #   '\nT_bin: %.1f' % Tbin)
        if plot:       
            fig.legend()
            
    return Irrad_array, beta_matrix, bethas

def fit_quotient_for_beta_twolasers(p, x, laser1, Tzero, Irrad1_n, Irrad1_m, Irrad2_n, Irrad2_m, y, error_y):
    quotient_lambda_func = lambda energy, beta1, beta2 : quotient_with_slope_twolasers(energy, laser1, Tzero, beta1, Irrad1_n, Irrad1_m, beta2, Irrad2_n, Irrad2_m)
    return curve_fit(quotient_lambda_func, x, y, p0 = p, bounds=(0, 5000), sigma=error_y, absolute_sigma=True)

def quotient_with_slope_twolasers(energy, laser1, Tzero, slope1, Irrad1_n, Irrad1_m, slope2, Irrad2_n, Irrad2_m):
    # Antistokes quotient fitting function
    # E is the energy array
    k = 0.000086173 # Boltzmann's constant in eV/K
    El = lambda_to_energy(laser1) # excitation wavelength in eV
    Tn = Tzero + slope1*Irrad1_n + slope2*Irrad2_n
    Tm = Tzero + slope1*Irrad1_m + slope2*Irrad2_m
    
  #  print('Irrad_n', Irrad_n, 'Irrad_m', Irrad_m)
    
    aux_n = (energy - El) / ( k*Tn )
    aux_m = (energy - El) / ( k*Tm )
    # two equivalent expressions can be used for the quotient
    # reduced form:
#    quotient = A * np.exp( -(aux_n - aux_m) ) * np.sinh(aux_m) / np.sinh(aux_n)
    # explicit form:
    bose_m = 1/( np.exp(aux_m) - 1 )
    bose_n = 1/( np.exp(aux_n) - 1 )
    quotient = (Irrad1_n * bose_n) / ( Irrad1_m * bose_m )
    return quotient


def Q_AS_two_lasers(long, laser1, size_notch, To, nbins, m_ref, noise, omega1, QY1, betha1, dTmax1, omega2, QY2, betha2, dTmax2):
    
   # print('laser', laser, 'betha', betha)
    
    desired_AS = np.where(long < laser1-int(size_notch/2) )
    long_AS = long[desired_AS]
   # desired_AS = np.where(long_AS > long_AS[0] + 7)
  #  long_AS = long_AS[desired_AS]
    
    energyAS = lambda_to_energy(long_AS)
    El = lambda_to_energy(laser1) # excitation wavelength in eV
    
    e = energyAS - El
    desired = np.where((e > 0.05) & (e < 0.09))
    energyAS = energyAS[desired]
    energy_shift = e[desired]
    
  #  e = energyAS - El
    
 #   desired = np.where(e < min(e) + 0.03)
 #   energyAS = energyAS[desired]
    
    r = np.linspace(0, np.sqrt(2*0.400**2), nbins)
    g1 = np.exp(-2*r**2/omega1**2)
    g1 = np.flip(g1)
    
    Imax1 = round(dTmax1/betha1,4)
    I_array1 = g1*Imax1
    dT_array1 = betha1*I_array1
    
    g2 = np.exp(-2*r**2/omega2**2)
    g2 = np.flip(g2)
    
    Imax2 = round(dTmax2/betha2,4)
    I_array2 = g2*Imax2
    dT_array2 = betha2*I_array2
    
    AS = np.zeros((nbins, len(energyAS)))
    errorAS = np.zeros((nbins, len(energyAS)))
    
    bkg =  np.ones(len(energyAS)) #np.random.normal(0,0.05,len(energyAS))
    
    plt.figure()
    #plt.xlabel('Wavelength (nm)')
    plt.xlabel('energy (eV)')
    plt.ylabel('I*BE')
  #  plt.ylim(1, 2.5)
    
    totalbins = nbins - 1
    
 #   print( 'total bins',   totalbins)
    
    ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    
    error_ydata_all = np.zeros((totalbins, totalbins, len(energyAS)))
    
    for i in range(nbins):
        
        dT1 = dT_array1[i]
        I1 = I_array1[i]
        
        dT2 = dT_array2[i]
        I2 = I_array2[i]
        
        T = dT1 + dT2 + To
        
        BE = bose(energyAS, El, T)
        signal = QY1*I1*BE + bkg
        
        signal = add_white_noise(signal, 0, noise)
        
        plt.plot(energy_shift, signal, '.', label = i)
        
        smooth_signal = savgol_filter(signal, 51, 1, mode = 'interp')
        
        AS[i, :] = smooth_signal
        
        error_signal = np.sqrt((smooth_signal-signal)**2)/2
        
        errorAS[i, :] = error_signal

        #signal = savgol_filter(signal, 51, 1, mode = 'interp')
        
        plt.plot(energy_shift, AS[i, :], 'k--')
        
    for k in range(totalbins):
        
        AS_ref = AS[k+1, :] - bkg
        errorAS_ref = errorAS[k+1, :]
    
       # plt.figure()
       # plt.title('ref: %s'%(k))
       # plt.xlabel('Energy (eV)')
       # plt.ylabel('Q')
            
        for j in range(totalbins):
            
            AS_j = AS[j+1, :]- bkg
            errorAS_j = errorAS[j+1, :]
            
            Q =  AS_j/AS_ref
            
            ydata_all[j,k,:] = Q
            
            errorQ = ( errorAS_j/AS_j + errorAS_ref/AS_ref )*Q
            
            error_ydata_all[j,k,:] = errorQ
            
          #  plt.plot(energyAS, Q, label = '%s,ref: %s'%(j,k))
        
      #  plt.legend()
        
    #plt.close('all')
   
    Irrad_array1 = I_array1[1:]
    err_Irrad_array1 = 0.01*Irrad_array1
    
    Irrad_array2 = I_array2[1:]
    err_Irrad_array2 = 0.01*Irrad_array2
            
    R2_matrix = np.zeros([totalbins, totalbins])
 #   chi_squared_matrix = np.zeros([totalbins, totalbins])
 #   p_value_matrix = np.zeros([totalbins, totalbins])
 #   beta_matrix = np.zeros([totalbins, totalbins])
 #   err_beta_matrix = np.zeros([totalbins, totalbins])
 #   Tzero_matrix = np.zeros([totalbins, totalbins])
 #   err_Tzero_matrix = np.zeros([totalbins, totalbins])
 #   T_matrix = np.zeros([totalbins, totalbins])
 #   err_T_matrix = np.zeros([totalbins, totalbins])
        
    skip_neighbours = 0
    x = energyAS 
    rango = list(range(0, totalbins))
    norm = plt.Normalize()
    colormap = plt.cm.plasma(norm(rango))
    
    Elaser = lambda_to_energy(laser1)
    #last_bin = list(range(0, totalbins))
    
    bethas2 = []
    bethas1 = []
    
    plot = False
           
    for ref in range(totalbins-1):
        print('\nFitting quotient Q using bin %d as reference...' % ref)
        
        #plt.figure(num=1,clear=True)
        list_of_bins_to_fit = list(range(ref+1+skip_neighbours, totalbins))
       # list_of_bins_to_fit = list(range(totalbins-1, ref+1+skip_neighbours-1, -1))
        
       # ax.set_ylim(0, 1) #50)
       
        if plot:
            fig, ax = plt.subplots()
            ax.set_xlim(0, 0.15)
            
        
        for i in list_of_bins_to_fit:
            
            if m_ref:
            
                # m = ref
                Irrad1_n = Irrad_array1[i]
                Irrad1_m = Irrad_array1[ref]
                err_Irrad1_n = err_Irrad_array1[i]
                err_Irrad1_m = err_Irrad_array1[ref]
                Irrad2_n = Irrad_array2[i]
                Irrad2_m = Irrad_array2[ref]
                err_Irrad2_n = err_Irrad_array2[i]
                err_Irrad2_m = err_Irrad_array2[ref]
                y = ydata_all[i, ref, :]
                error_y = error_ydata_all[i,ref, :]
            
            else:
                # n = ref
                Irrad1_m = Irrad_array1[i]
                Irrad1_n = Irrad_array1[ref]
                Irrad2_m = Irrad_array2[i]
                Irrad2_n = Irrad_array2[ref]
          #      err_Irrad1_m = err_Irrad_array1[i]
          #      err_Irrad1_n = err_Irrad_array1[ref]
          #      err_Irrad2_m = err_Irrad_array2[i]
          #      err_Irrad2_n = err_Irrad_array2[ref]
                y = ydata_all[ref,i, :]
                error_y = error_ydata_all[ref,i, :]
            
           # print('irradiance', Irrad_n, Irrad_m, y)
            
            Tzero = To
            init_params = betha1, betha2
            best_as, err = fit_quotient_for_beta_twolasers(init_params, x, laser1, Tzero, Irrad1_n, Irrad1_m, Irrad2_n, Irrad2_m, y, error_y)
                # retrieve best fitted parameters
            b1 = best_as[0]
            b2 = best_as[1]
            
         #   Tref = temp_with_slope(Tzero, b, Irrad_m)
         #   Tbin = temp_with_slope(Tzero, b, Irrad_n)
            
                # calculate the errors and goodes of the fit
         ##   err_beta = np.sqrt(err[0,0])
           # err_Tzero = 0.5
           # err_Tref = np.sqrt(err_Tzero**2 + (Irrad_m*err_beta)**2 + (err_Irrad_m*b)**2)
           # err_Tbin = np.sqrt(err_Tzero**2 + (Irrad_n*err_beta)**2 + (err_Irrad_n*b)**2)
            yfit = quotient_with_slope_twolasers(x, laser1, Tzero, b1, Irrad1_n, Irrad1_m, b2, Irrad2_n, Irrad2_m)
            r2_coef_pearson = calc_r2(y, yfit)
     #       chi_squared_pearson = calc_chi_squared(y, yfit)
                # Plotting
            
            if plot:
            
                color_iter = colormap[i]
                ax.plot(x-Elaser, y, '-', linewidth = 2.0, color = color_iter, label='$Q_{%d/%d}$' % (i, ref))
                ax.plot(x-Elaser, yfit, '--k', alpha = 0.8)
          
                # Asign matrix elements
       #     beta1_matrix[ref,i] = b1
       #     err_beta1_matrix[ref,i] = err_beta1
            
         #   Tzero_matrix[ref,i] = Tzero
         #   err_Tzero_matrix[ref,i] = err_Tzero
         ##   T_matrix[ref,i] = Tref
           # T_matrix[i,ref] = Tbin
           # err_T_matrix[ref,i] = err_Tref
           # err_T_matrix[i,ref] = err_Tbin
            R2_matrix[ref,i] = r2_coef_pearson
            R2_matrix[i,ref] = r2_coef_pearson
            
            if r2_coef_pearson > 0.8:
                bethas1.append(b1)
                bethas2.append(b2)
            
        #    chi_squared_matrix[ref,i] = chi_squared_pearson
        #    chi_squared_matrix[i,ref] = chi_squared_pearson
        #    p_value_matrix[ref,i] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    p_value_matrix[i,ref] = 1 - frozen_distro.cdf(chi_squared_pearson)
        #    print('---------- Bin', i, \
          #            '\nR-sq: %.3f' % r2_coef_pearson, \
          #            '\nTzero: %.1f' % Tzero, 'err Tzero: %.1f' % err_Tzero, \
            #          '\nbeta: %.1f' % beta, 'error beta: %.2f' % err_beta, \
               #       '\nT_ref: %.1f' % Tref, 'error T_ref: %.1f' % err_Tref, \
               #       '\nT_bin: %.1f' % Tbin, 'error T_bin: %.1f' % err_Tbin)
               
          #  print('---------- Bin', i, \
                #      '\nR-sq: %.3f' % r2_coef_pearson, \
                #      '\nTzero: %.1f' % Tzero,  \
                 #     '\nbeta: %.1f' % b,  \
                 #     '\nT_ref: %.1f' % Tref, \
                   #   '\nT_bin: %.1f' % Tbin)
        if plot:       
            fig.legend()
            
    return Irrad_array1, bethas1, Irrad_array2, bethas2


#%%

plt.close('all')

size_notch = 20
laser = 532 #532
long = np.arange(laser-30, laser+70, 0.1)
#betha = 100 #60
Tmax = 400
To = 293
nbins = 10
omega = 0.340
mref = False
QY = 1

#n = round((Tmax+dT-To)/dT)
totals = nbins - 1
Ntotal = float((totals**2-totals)/2)

sigma2 = 36
ks = np.array([0.1, 0.6])
betha_array = sigma2/ks 

noise_array = np.array([0.001])

fig1, ax1  = plt.subplots()
ax1.plot(betha_array, betha_array, 'k--')

fig2, ax2  = plt.subplots()
ax2.set_ylim(-0.1, 1.1)

for i in range(len(noise_array)):
    
    b_array1 = []
    b_err_array1 = []
    N_array1 = []
    
    noise = noise_array[i]

    for betha in betha_array:
    
        Irrad_array, beta_matrix, bethas = Q_AS(long, laser, size_notch, QY, betha, Tmax, To, nbins, omega, mref, noise)
        
        print('laser', laser)
        print('irradiance', Irrad_array)
        print('N betha ok', len(bethas))
        print('betha', np.round(np.median(bethas), 4), np.round(np.std(bethas),4))
        
        p = 100*np.round(np.std(bethas),4)/np.round(np.median(bethas), 4)
        print('precision %', p)
        print('exactitud', betha - np.round(np.median(bethas),4) )
        
        b_array1.append(np.round(np.median(bethas), 4))
        b_err_array1.append(np.round(np.std(bethas)/np.sqrt(len(bethas)),4))
        N_array1.append(len(bethas))
    
    
    N_array1 = np.array(N_array1)
    
    ax1.errorbar(betha_array, b_array1, yerr = b_err_array1, fmt = 'o')

    ax2.plot(ks, N_array1/Ntotal, 'o')
   

#%%

QY = 1
laser = 642 #532
long = np.arange(laser-30, laser+70, 0.1)

b_array = []
b_err_array = []
N_array = []

for betha in betha_array:
    Irrad_array, beta_matrix, bethas = Q_AS(long, laser, size_notch, QY, betha, Tmax, To, nbins, omega, mref, noise)
    
    print('laser', laser)
    #print('irradiance', Irrad_array)
    print('N betha ok', len(bethas))
    print('betha', np.round(np.median(bethas), 4), np.round(np.std(bethas),4))
    
    p = 100*np.round(np.std(bethas),4)/np.round(np.median(bethas), 4)
    print('precision %', p)
    print('exactitud', betha - np.round(np.median(bethas),4) )
    
    b_array.append(np.round(np.median(bethas), 4))
    b_err_array.append(np.round(np.std(bethas)/np.sqrt(len(bethas)),4))
    N_array.append(len(bethas))

    
N_array = np.array(N_array)

fig1, ax1  = plt.subplots()
ax1.errorbar(betha_array, b_array1, yerr = b_err_array1, fmt = 'o')
ax1.plot(betha_array, betha_array, '--')
        

fig2, ax2   = plt.subplots()
ax2.plot(betha_array, N_array1/Ntotal, 'o')

ax1.errorbar(betha_array, b_array, yerr = b_err_array, fmt = '*', color = 'C2')
ax2.plot(betha_array, N_array/Ntotal, '*', color = 'C2')
ax2.set_ylim(0, 1.1)

#%%

plt.close('all')

QY = 1

power_bfp_max_array = np.array([0.5, 0.6, 0.8]) #mW
omega = 0.340   #np.array([0.340, 0.5, 0.6]) #um2
tr = 0.47
Imax_array = 2*tr*power_bfp_max_array/(np.pi*omega**2)

print('irradiance', Imax_array)

nbins = 10
size_notch = 20
laser = 532 #532
long = np.arange(laser-30, laser+70, 0.1)
#betha = 100 #60
To = 293
mref = False
noise = 0.001 #0.001

totals = nbins - 1
Ntotal = float((totals**2-totals)/2)

betha = 60

b_array1 = []
b_err_array1 = []
N_array1 = []

for i in range(len(Imax_array)):
    
    Imax = Imax_array[i]

    Irrad_array, beta_matrix, bethas = Q_AS_irradiance(long, laser, size_notch, QY, betha, Imax, To, nbins, omega, mref, noise)
    
    print('temperature max', max(Irrad_array)*betha + To)
    
    print('laser', laser)
    #print('irradiance', Irrad_array)
    print('N betha ok', len(bethas))
    print('betha', np.round(np.median(bethas), 4), np.round(np.std(bethas),4))
    
    p = 100*np.round(np.std(bethas),4)/np.round(np.median(bethas), 4)
    print('precision %', p)
    print('exactitud', betha - np.round(np.median(bethas),4) )
    
    b_array1.append(np.round(np.median(bethas), 4))
    b_err_array1.append(np.round(np.std(bethas)/np.sqrt(len(bethas)),4))
    N_array1.append(len(bethas))
    
   # plt.close('all')
    
N_array1 = np.array(N_array1)

fig1, ax1  = plt.subplots()
ax1.errorbar(Imax_array, b_array1, yerr = b_err_array1, fmt = 'o')
#ax1.plot(betha_array, betha_array, '--')

fig2, ax2   = plt.subplots()
ax2.plot(Imax_array, N_array1/Ntotal, 'o')
ax2.set_ylim(0, 1.1)

#%%

plt.close('all')

nbins = 10
totals = nbins - 1
Ntotal = float((totals**2-totals)/2)

size_notch = 20
laser1 = 532 #532
long = np.arange(laser1-30, laser1+70, 0.1)
#betha = 100 #60

To = 293
mref = False
noise = 0.001 #.001 # 0.001 #0.001

Tmax_total = 400

print('dTmax total', Tmax_total - To)

QY1 = 1    ### excitación
QY2 = 0    ### calentador

omega1 = 0.340
omega2 = 0.340

sigma1 = 36
ks = np.array([0.1, 0.6])

betha1_array = sigma1/ks  #np.array([100, 60])

print('betha1', betha1_array)

f = 1/10

betha2_array = betha1_array*f

print('betha2', betha2_array)

fig1, ax1  = plt.subplots()
ax1.plot(betha2_array, betha2_array, 'k--')
ax1.plot(betha1_array, betha1_array, 'r--')

fig2, ax2  = plt.subplots()
ax2.set_ylim(-0.1, 1.1)

dTmax1_array = [40]

for dTmax1 in dTmax1_array:
    
    b_array1 = []
    b_err_array1 = []
    
    b_array2 = []
    b_err_array2 = []
    
    N_array = []

    dTmax2 = Tmax_total - To - dTmax1
    
    for betha1 in betha1_array:
        
    
        betha2 = f*betha1
    
        Irrad_array1, bethas1, Irrad_array2, bethas2 = Q_AS_two_lasers(long, laser1, size_notch, To, nbins, mref, noise, omega1, QY1, betha1, dTmax1, omega2, QY2, betha2, dTmax2)
    
        print('LASER 2 HEAT, LASER 1 EXCITED')
        print('irradiance 1', Irrad_array1)
        print('irradiance 2', Irrad_array2)
        print('N betha2 ok', len(bethas2))
        print('betha1', np.round(np.median(bethas1), 4), np.round(np.std(bethas1),4))
        
        print('betha2', np.round(np.median(bethas2), 4), np.round(np.std(bethas2),4))
        p = 100*np.round(np.std(bethas2),4)/np.round(np.median(bethas2), 4)
        print('precision %', p)
        print('exactitud', betha2 - np.round(np.median(bethas2),4) )
        
        b_array1.append(np.round(np.median(bethas1), 4))
        b_err_array1.append(np.round(np.std(bethas1)/np.sqrt(len(bethas1)),4))
        
        b_array2.append(np.round(np.median(bethas2), 4))
        b_err_array2.append(np.round(np.std(bethas2)/np.sqrt(len(bethas2)),4))
        
        N_array.append(len(bethas2))
        
    N_array = np.array(N_array)
        
    ax1.errorbar(betha2_array, b_array2, yerr = b_err_array2, fmt = 'o')
    ax1.errorbar(betha1_array, b_array1, yerr = b_err_array1, fmt = 'o')
    ax2.plot(ks, N_array/Ntotal, 'o')
       
