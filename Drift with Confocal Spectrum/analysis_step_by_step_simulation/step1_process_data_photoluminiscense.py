# -*- coding: utf-8 -*-
"""
Process of single AuNPs spectral confocal image 
acquired with PySpectrum at CIBION

Mariano Barella

16 aug 2019

based on "witec_data_photoluminiscense.py"

"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar
import scipy.signal as sig
import scipy.stats as sta
from functions_for_photoluminiscence import manage_save_directory, \
    classification, lambda_to_energy, gaussian, fit_gaussian, quotient, \
    fit_quotient, calc_r2, calc_chi_squared, closer, lorentz2, fit_lorentz2

try:
    plt.style.use('for_confocal.mplstyle')
except:
    print('Pre-defined matplotlib style was not loaded.')

plt.ioff()
plt.close('all')

def process_confocal_to_bins(folder, path_to, totalbins, image_size_px, 
                             image_size_um, camera_px_length, window, deg, 
                             repetitions, factor, meas_pow_bfp, start_notch, 
                             end_notch, end_power, start_spr, lower_londa, 
                             upper_londa, plot_flag=False):
    
    NP = folder.split('Spectrum_')[-1]
    
    save_folder = os.path.join(path_to, NP)
    
    common_path = os.path.join(path_to,'common_plots')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
            
    list_of_files = os.listdir(folder)
    wavelength_filename = [f for f in list_of_files if re.search('wavelength',f)]
    list_of_files.sort()
    list_of_files = [f for f in list_of_files if not os.path.isdir(folder+f)]
    list_of_files = [f for f in list_of_files if ((not os.path.isdir(folder+f)) \
                                                  and (re.search('_i\d\d\d\d_j\d\d\d\d.txt',f)))]
    L = len(list_of_files)            
    
    data_spectrum = []
    name_spectrum = []
    specs = []
        
    print(L, 'spectra were acquired.')
    
    for k in range(L):
        name = os.path.join(folder,list_of_files[k])
        data_spectrum = np.loadtxt(name)
        name_spectrum.append(list_of_files[k])
        specs.append(data_spectrum)
    
    wavelength_filepath = os.path.join(folder,wavelength_filename[0])
    londa = np.loadtxt(wavelength_filepath)
    
    start_notch = closer(londa, start_notch)
    end_notch = closer(londa, end_notch)
    end_power = closer(londa, end_power)
    upper_londa_index = closer(londa, upper_londa)
    
    # ALLOCATING
    matrix_spec_raw = np.zeros((image_size_px,image_size_px,camera_px_length))
    matrix_spec_smooth = np.zeros((image_size_px,image_size_px,camera_px_length))
    matrix_spec = np.zeros((image_size_px,image_size_px,camera_px_length))
    matrix_spec_normed = np.zeros((image_size_px,image_size_px,camera_px_length))
    pixel_power = np.zeros((image_size_px,image_size_px))
    binned_power = np.zeros((image_size_px,image_size_px))
   
    for i in range(image_size_px):
        for j in range(image_size_px):
            matrix_spec_raw[i,j,:] = np.array(specs[i*image_size_px+j])
    del specs
    
    ######################## SMOOTH ############################
    ######################## SMOOTH ############################
    ######################## SMOOTH ############################
    
    # SPLIT SIGNALS INTO STOKES AND ANTI-STOKES
    matrix_stokes_raw = matrix_spec_raw[:,:,end_notch:]
    londa_stokes = londa[end_notch:]
    
    matrix_antistokes_raw = matrix_spec_raw[:,:,:upper_londa_index]
    londa_antistokes = londa[:upper_londa_index]
    
    # SMOOTHING
    print('Smoothing signals...')
    aux_matrix_stokes_smooth = sig.savgol_filter(matrix_stokes_raw, 
                                               window, deg, axis = 2, 
                                               mode='mirror')
    aux_matrix_antistokes_smooth = sig.savgol_filter(matrix_antistokes_raw, 
                                               window, deg, axis = 2, 
                                               mode='mirror')
    
    for i in range(repetitions-1):
        aux_matrix_stokes_smooth = sig.savgol_filter(aux_matrix_stokes_smooth,
                                                   window, deg, axis = 2, 
                                                   mode='mirror')
        aux_matrix_antistokes_smooth = sig.savgol_filter(aux_matrix_antistokes_smooth,
                                                   window, deg, axis = 2, 
                                                   mode='mirror')
    # Merge
    matrix_stokes_smooth = aux_matrix_stokes_smooth
    matrix_antistokes_smooth = aux_matrix_antistokes_smooth
    matrix_spec_smooth[:,:,end_notch:] = matrix_stokes_smooth
    matrix_spec_smooth[:,:,:upper_londa_index] = matrix_antistokes_smooth
    
    if plot_flag:
        print('Saving single spectrums as measured...')
        for i in range(image_size_px):
            for j in range(image_size_px):
                plt.figure()
                pixel_name = 'i%02d_j%02d' % (i,j)
                plt.plot(londa, matrix_spec_raw[i,j,:], color='C0', linestyle='-', label='As measured')
                plt.plot(londa_stokes, matrix_stokes_smooth[i,j,:], color='k', linestyle='-', label='Smoothed')
                plt.plot(londa_antistokes, matrix_antistokes_smooth[i,j,:], color='k', linestyle='-')
                plt.legend()
                ax = plt.gca()
                ax.set_xlabel(r'Wavelength (nm)')
                ax.set_ylabel('Photoluminiscence (a.u.)')
                ax.axvline(londa[start_notch], ymin = 0, ymax = 1, color='k', linestyle='--')
                ax.axvline(londa[end_notch], ymin = 0, ymax = 1, color='k', linestyle='--')
                ax.axvline(londa[end_power], ymin = 0, ymax = 1, color='k', linestyle='--')
                aux_folder = manage_save_directory(save_folder,'pl_spectra_as_measured')
                figure_name = os.path.join(aux_folder, 'spec_%s_%s.png' % (pixel_name, NP))
                ax.set_ylim([7000,20000])
                plt.savefig(figure_name, dpi=100)
                plt.close()
    
    ######################## KILL NOTCH RANGE ############################
    ######################## KILL NOTCH RANGE ############################
    ######################## KILL NOTCH RANGE ############################

    matrix_spec_smooth[:,:,start_notch:end_notch] = np.nan
    
    ######################## FIND BKG AND MAX ############################
    ######################## FIND BKG AND MAX ############################
    ######################## FIND BKG AND MAX ############################
    
    # LOOK FOR MAX AND MIN IN STOKES RANGE
    aux_sum = np.sum(matrix_stokes_smooth, axis=2)
    
    print('Finding max and bkg...')            
    
    imin, jmin = np.unravel_index(np.argmin(aux_sum, axis=None), aux_sum.shape)
    bkg_smooth = matrix_spec_smooth[imin, jmin, :]
    bkg_raw = matrix_spec_raw[imin, jmin, :]
    noise_rms = np.nansum(bkg_raw**2)
    
    imax, jmax = np.unravel_index(np.argmax(aux_sum, axis=None), aux_sum.shape)
    max_smooth = matrix_spec_smooth[imax, jmax, :]
    max_raw = matrix_spec_raw[imax, jmax, :]
    signal_rms = np.nansum(max_smooth**2)
    
    signal_to_background_ratio = (signal_rms/noise_rms)**2
    print('Signal to bkg ratio:', signal_to_background_ratio)
    
    signal_to_background = max_raw/bkg_raw
    # signal_to_background SPECTRUM
    plt.figure()
    plt.plot(londa, signal_to_background)
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    figure_name = os.path.join(save_folder,'signal_to_background_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    # BACKGROUND SUBSTRACTION
    matrix_spec = matrix_spec_smooth - bkg_smooth
                
    # BACKGROUND SPECTRUM
    plt.figure()
    plt.plot(londa, bkg_raw)
    plt.plot(londa, bkg_smooth, '-k')
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    ax.set_ylim([8000, 20000])
    figure_name = os.path.join(save_folder,'bkg_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'bkg')
    figure_name = os.path.join(aux_folder,'bkg_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    # MAX SPECTRUM
    plt.figure()
    plt.plot(londa, max_raw)
    plt.plot(londa, max_smooth, '-k')
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    ax.set_ylim([8000, 20000])
    figure_name = os.path.join(save_folder,'max_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'max')
    figure_name = os.path.join(aux_folder,'max_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()

    if plot_flag:
        print('Saving single spectrums without bkg...')
        for i in range(image_size_px):
            for j in range(image_size_px):
                plt.figure()
                pixel_name = 'i%02d_j%02d' % (i,j)
                plt.plot(londa, matrix_spec[i,j,:], label=pixel_name)
                plt.legend()
                ax = plt.gca()
                ax.set_xlabel(r'Wavelength (nm)')
                ax.set_ylabel('Photoluminiscence (a.u.)')
                aux_folder = manage_save_directory(save_folder,'pl_spectra_minus_bkg')
                figure_name = os.path.join(aux_folder,'spec_minus_bkg_%s_%s.png' % (pixel_name, NP))
                ax.set_ylim([0,10000])
                plt.savefig(figure_name)
                plt.close()  

    ######################## FITTING LORENTZ CURVE AT STOKES ############################
    ######################## FITTING LORENTZ CURVE AT STOKES ############################
    ######################## FITTING LORENTZ CURVE AT STOKES ############################

    print('Extracting SPR peak and width...')
    sum_stokes = np.sum(matrix_spec[:,:,end_notch:], axis = (0, 1))
    sum_stokes = sum_stokes/np.max(sum_stokes)
    # Fitting the data
    # initial parameter guesses
    # [amplitude, FWHM, center, offset]
    init_params = np.array([1, 50, 550, 0.05], dtype=np.double)
    # Get the fitting parameters for the best lorentzian
    best_lorentz, err = fit_lorentz2(init_params, londa_stokes, sum_stokes)
    # calculate the errors
    lorentz_fitted = lorentz2(londa_stokes, *best_lorentz)
    r2_coef_pearson = calc_r2(sum_stokes, lorentz_fitted)
    full_lorentz_fitted = lorentz2(londa, *best_lorentz)
    londa_max_pl = best_lorentz[2]
    width_pl = best_lorentz[1]
    
    # STOKES FITTING
    plt.figure()
    plt.plot(londa_stokes, sum_stokes,'C0', label='Data %s' % NP)
    plt.plot(londa, full_lorentz_fitted,'k--', label='Lorentz fit')
    plt.legend()
    ax = plt.gca()
    ax.set_ylabel(r'Intensity (a.u.)')
    ax.set_ylabel('SPR (a.u.)')
    plt.ylim([0,1.05])
    aux_folder = manage_save_directory(save_folder,'spr')
    figure_name = os.path.join(aux_folder, 'sum_specs_mean_fitted_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'spr')
    figure_name = os.path.join(aux_folder, 'sum_specs_mean_fitted_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close() 

    ######################## NORMALIZE STOKES ############################
    ######################## NORMALIZE STOKES ############################
    ######################## NORMALIZE STOKES ############################

    print('Normalize stokes...')

    # MAKE MATRIX OF POWER
    matrix_spec_normed = matrix_spec / ( max_smooth - bkg_smooth )

    plt.figure()
    for i in range(image_size_px):
        for j in range(image_size_px):
            pixel_name = 'i%02d_j%02d' % (i,j)
            plt.plot(londa, matrix_spec_normed[i,j,:], label=pixel_name)
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    ax.axvline(londa[end_notch], ymin = -1, ymax = 2, color ='k', linestyle='--')
    ax.axvline(londa[end_power], ymin = -1, ymax = 2, color ='k', linestyle='--')
    ax.set_ylim([-0.1,1.1])
    ax.set_xlim([0.99*londa[end_notch],1.01*londa[end_power]])
    aux_folder = manage_save_directory(save_folder,'pl_stokes_normalized')
    figure_name = os.path.join(aux_folder,'spec_stokes_normalized_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'pl_stokes_normalized')
    figure_name = os.path.join(aux_folder,'spec_stokes_normalized_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    ######################## FIND POWER AND IRRADIANCE ############################
    ######################## FIND POWER AND IRRADIANCE ############################
    ######################## FIND POWER AND IRRADIANCE ############################
    
    print('Performing pixel power ratio...')
    # A PARTIR DEL NOTCH, UNA ZONA FLAT CALCULO POTENCIA
    pixel_power = np.mean(matrix_spec_normed[:,:,end_notch:end_power],axis=2)
    
    plt.figure()
    ax = plt.gca()
    ax.set_xticks(range(image_size_px))
    ax.set_yticks(range(image_size_px))
    img = ax.imshow(pixel_power, interpolation='none', cmap=cm.gist_gray)
    cbar = plt.colorbar(img, ax=ax)
    figure_name = os.path.join(save_folder,'pixel_power_%s.png' % NP)
    plt.grid(False)
    plt.savefig(figure_name)
    plt.close()
    
    ######################## PSF GAUSSIANA ############################
    ######################## PSF GAUSSIANA ############################
    ######################## PSF GAUSSIANA ############################
    
    print('Fitting PSF...')
    pixel_size = image_size_um/image_size_px # in um^2
    x = np.arange(-image_size_um/2, image_size_um/2, pixel_size)
    x_long = np.arange(-1.5, 1.5, 0.01)
    # Fitting the data
    # initial parameter guesses
    # [amplitude, w0, center]
    init_params = np.array([0.8, 0.3, 0], dtype=np.double)
    # Get the fitting parameters for the best gaussian in X
    power_x = np.sum(pixel_power, axis=0)
    best_x, err_x = fit_gaussian(init_params, x, power_x)
    # calculate the errors
    xfit = gaussian(x, *best_x)
    xfit2 = gaussian(x_long, *best_x)
    r2_coef_pearson_x = calc_r2(power_x, xfit)
    w0x = np.abs(best_x[1])
    err_w0x = np.sqrt(err_x[1,1])
    FWHMx = w0x*np.sqrt(2*np.log(2))
    sigmax = w0x/2
    # Get the fitting parameters for the best gaussian in Y
    power_y = np.sum(pixel_power,axis=1)
    best_y, err_y = fit_gaussian(init_params, x, power_y)
    # calculate the errors
    yfit = gaussian(x, *best_y)
    yfit2 = gaussian(x_long, *best_y)
    r2_coef_pearson_y = calc_r2(power_y, yfit)
    w0y = np.abs(best_y[1])
    err_w0y = np.sqrt(err_y[1,1])
    FWHMy = w0y*np.sqrt(2*np.log(2))
    sigmay = w0y/2
    
    w0 = (w0x + w0y)/2
    err_w0 = (err_w0x + err_w0y)/2
    
    print('Fitted parameters and waists (in um):')
    print('R2-x-y: ', r2_coef_pearson_x, r2_coef_pearson_y)
    print('w0x: ', w0x, 'FWHMx: ', FWHMx, 'sigmax: ', sigmax)
    print('w0y: ', w0y, 'FWHMy: ', FWHMy, 'sigmay: ', sigmay)
    print('Mean w0: ', w0, 'err: ', err_w0)

    # POWER PROFILE IN X AND Y
    plt.figure()
    plt.plot(x,power_x, 'o-', color='C0', markersize=8, linewidth=2, label='x')
    plt.plot(x,power_y, 's-', color='C1', markersize=8, linewidth=2, label='y')
    plt.plot(x_long,xfit2, '--', color='k', linewidth=2, label='Gaussian fit')
    plt.plot(x_long,yfit2, '--', color='k', linewidth=2)
    plt.legend(loc='best')
    ax = plt.gca()
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Power (mW)')
    ax.set_xlim([-image_size_um/2,image_size_um/2])
    aux_folder = manage_save_directory(save_folder,'psf')
    figure_name = os.path.join(aux_folder,'psf_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()

    ######################## IRRADIANCE ############################
    ######################## IRRADIANCE ############################
    ######################## IRRADIANCE ############################
    
    print('Calc irradiance...')            

    meas_pow_sample = factor*meas_pow_bfp
    irrad_calc = 2*meas_pow_sample/(np.pi*w0**2)
    
    print('===> Irradiance max (calc): %.2f mW/um2\n' % irrad_calc)
    
    total_power = np.sum(pixel_power)
    pixel_irrad_alt = meas_pow_sample*(pixel_power/total_power)/pixel_size**2
    
    pixel_irrad = irrad_calc*pixel_power

    # IMAGE POWER HOT MAP
    plt.figure()
    img = plt.imshow(pixel_irrad*1000.0, interpolation='none', cmap='hot')
    ax = plt.gca()
    ax.set_xticks(range(image_size_px))
    ax.set_yticks(range(image_size_px))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    scalebar = ScaleBar(pixel_size*1e-6)
    ax.add_artist(scalebar)
    cbar = plt.colorbar()
    cbar.ax.set_title(u'Irradiance (µW/µm$^{2}$)', fontsize=13)
    figure_name = os.path.join(save_folder,'pixel_irrad_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'pixel_irrad')
    figure_name = os.path.join(aux_folder,'pixel_irrad_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
#    plt.show()

#    # IMAGE POWER HOT MAP ALTERNATIVE
    plt.figure()
    img = plt.imshow(pixel_irrad_alt*1000.0, interpolation='none', cmap='hot')
    ax = plt.gca()
    ax.set_xticks(range(image_size_px))
    ax.set_yticks(range(image_size_px))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    scalebar = ScaleBar(pixel_size*1e-6) 
    ax.add_artist(scalebar)
    cbar = plt.colorbar()
    cbar.ax.set_title(u'Irradiance (µW/µm$^{2}$)', fontsize=13)
    figure_name = os.path.join(save_folder,'pixel_irrad_alternative_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'pixel_irrad')
    figure_name = os.path.join(aux_folder,'pixel_irrad_alternative_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    ######################## BINNING ############################
    ######################## BINNING ############################
    ######################## BINNING ############################
    
    print('Binning...')
    
    for i in range(image_size_px):
        for j in range(image_size_px):
            nbin = classification(pixel_power[i,j], totalbins, [0,1])
            binned_power[i,j] = nbin
    
    plt.figure()
    ax = plt.gca()
    ax.set_xticks(range(image_size_px))
    ax.set_yticks(range(image_size_px))
    rango = range(0,totalbins+1)
    ticks_label_rango = [float(i)/totalbins for i in rango]
    img = ax.imshow(binned_power, interpolation='none', cmap=cm.gist_gray)
    cbar = plt.colorbar(img, ax=ax, boundaries=rango, ticks=rango)
    cbar.ax.set_yticklabels(ticks_label_rango)  # vertically oriented colorbar
    plt.grid(False)
    figure_name = os.path.join(save_folder,'binned_power_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'binned_power')
    figure_name = os.path.join(aux_folder,'binned_power_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    # SPECTRUMS IN SAME GRAPH ACCORDING TO ITS POWER BIN
    mean_specs = np.zeros([totalbins,camera_px_length])
    mean_pow = np.zeros(totalbins)
    mean_irrad = np.zeros(totalbins)
    hist_bin = np.zeros(totalbins)
    for s in range(totalbins):
        u, v = np.where(binned_power == s)
        counter = 0
        aux_spec = np.zeros(camera_px_length)
        aux_pow = 0
        aux_irrad = 0
        plt.figure()
        for i, j in zip(u,v):
            plt.plot(londa,matrix_spec[i,j,:])
            aux_pow += pixel_power[i,j]
            aux_irrad += pixel_irrad[i,j]
            aux_spec += matrix_spec[i,j,:]
            counter += 1
        if counter == 0:
            print('Bin %d has NO spectra to average.' % s)
            aux_spec = aux_spec
            aux_pow = aux_pow
            aux_irrad = aux_irrad
        else:
            aux_spec = aux_spec/counter
            aux_pow = aux_pow/counter
            aux_irrad = aux_irrad/counter
        mean_specs[s] = aux_spec
        mean_pow[s] = aux_pow
        mean_irrad[s] = aux_irrad
        hist_bin[s] = counter
        plt.plot(londa,aux_spec, '--k')
        plt.title('Bin %d' % s)
        plt.ylim([0,10000])
        aux_folder = manage_save_directory(save_folder,'pl_in_bins')
        figure_name = os.path.join(aux_folder,'bin_%d_%s.png' % (s, NP))
        plt.savefig(figure_name)
        plt.close()
    
    # binning error determination
    estimated_value = hist_bin/np.sum(hist_bin)
    err_bin = np.round(np.sqrt(hist_bin*(1-estimated_value))) # binomial std deviation
    
    # find theorical binning
    edges = np.array(range(totalbins))
    centers = edges + 0.5
    centers_distro = centers[::-1]
    height_distro = np.zeros(totalbins)
    step = 1/totalbins
    binned_range = np.arange(0, 1 + step, step)[::-1]
    for i in range(totalbins-1):
        I_1 = binned_range[i]
        I_2 = binned_range[i+1]
        height_distro[i] = np.round(0.5*(w0**2)*np.log(I_1/I_2)*np.pi/(pixel_size**2))
#        print(height_distro[i], 'between', I_1, I_2)
    height_distro[-1] = image_size_px**2 - np.sum(height_distro)
    
    # plot histogram of binning
    plt.figure()
    plt.bar(centers, hist_bin, align='center')
    plt.errorbar(centers, hist_bin, fmt='', linestyle='', yerr=err_bin)
    plt.plot(centers_distro, height_distro, 'o-', color='C3')
    plt.xlabel('Bin')
    plt.ylabel('Number of spectra')
    plt.ylim([0,image_size_px**2])
    aux_folder = manage_save_directory(save_folder,'pl_in_bins')
    figure_name = os.path.join(aux_folder,'hist_of_binning_%s_lin.png' % (NP))
    plt.savefig(figure_name)
    plt.ylim([0.9,image_size_px**2])
    ax = plt.gca()
    ax.set_yscale('log')
    aux_folder = manage_save_directory(save_folder,'pl_in_bins')
    figure_name = os.path.join(aux_folder,'hist_of_binning_%s_log.png' % (NP))
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'hist_of_binning')
    figure_name = os.path.join(aux_folder,'hist_of_binning_%s_log.png' % (NP))
    plt.savefig(figure_name)
    plt.close()
    
    penalization_array = np.zeros(totalbins)
    for s in range(totalbins):
        bin_count = hist_bin[s]
        err_bin_count = err_bin[s]
        est_value = height_distro[::-1][s]
#        print('bin count', bin_count)
#        print('est', est_value)
#        print('error', err_bin_count)
        if est_value <= (bin_count + err_bin_count) and \
            est_value >= (bin_count - err_bin_count):
            penalization = 0
        else:
            penalization = 1
            print('Bin %s has been penalized.' % s)
        penalization_array[s] = penalization
    
    ################### LAST BIN IS BKG ########################
    ################### LAST BIN IS BKG ########################
    ################### LAST BIN IS BKG ########################
    
    print('Last bin is bkg...')
    print('Correction to bin spectra is being applied...')
    
    # correct signals to account for mean bkg (last bin)
    corrected_mean_specs = mean_specs - mean_specs[0]
    corrected_mean_pow = mean_pow - mean_pow[0]
    corrected_mean_irrad = mean_irrad - mean_irrad[0]

    plt.figure()
    for i in range(totalbins):
        plt.plot(londa,corrected_mean_specs[i],label='Bin %d' % i)
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    plt.ylim([0,10000])
    plt.legend(loc='upper left')
    aux_folder = manage_save_directory(save_folder,'pl_in_bins')
    figure_name = os.path.join(aux_folder,'all_bins_corrected_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'all_bins')
    figure_name = os.path.join(aux_folder,'all_bins_corrected_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    print('Saving processed data...')
    
    aux_folder = manage_save_directory(save_folder,'pl_in_bins')
    to_save = corrected_mean_specs.T
    path_to_save = os.path.join(aux_folder,'all_bins_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
    
    to_save = penalization_array 
    path_to_save = os.path.join(aux_folder,'bin_penalization_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.0f')
        
    to_save = londa
    path_to_save = os.path.join(aux_folder,'londa_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
    
    to_save = corrected_mean_pow
    path_to_save = os.path.join(aux_folder,'bin_power_quotient_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
        
    to_save = corrected_mean_irrad
    path_to_save = os.path.join(aux_folder,'bin_irradiance_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
    
    aux_folder = manage_save_directory(save_folder,'spr')
    to_save = [londa_max_pl, width_pl, r2_coef_pearson]
    path_to_save = os.path.join(aux_folder,'spr_fitted_parameters_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
    
    aux_folder = manage_save_directory(save_folder,'psf')
    to_save = [w0x, err_w0x, r2_coef_pearson_x, w0y, err_w0y, r2_coef_pearson_y, w0, err_w0]
    path_to_save = os.path.join(aux_folder,'psf_fitted_parameters_%s.dat' % NP)
    np.savetxt(path_to_save, to_save, fmt='%.3e')
    
    return

def calculate_quotient(folder, path_to, totalbins, lower_londa, 
                       upper_londa, Tzero):
    
    NP = folder.split('Spectrum_')[-1]
    
    common_path = os.path.join(path_to,'common_plots')
    
    save_folder = os.path.join(path_to, NP)
    
    bin_folder = os.path.join(save_folder, 'pl_in_bins')
    
    londa_file = os.path.join(bin_folder,'londa_%s.dat' % NP)
    corrected_mean_specs_file = os.path.join(bin_folder,'all_bins_%s.dat' % NP)
    corrected_mean_pow_file = os.path.join(bin_folder,'bin_power_quotient_%s.dat' % NP)
    corrected_mean_irrad_file = os.path.join(bin_folder,'bin_irradiance_%s.dat' % NP)
    bin_penalization_file = os.path.join(bin_folder,'bin_penalization_%s.dat' % NP)
    
    londa = np.loadtxt(londa_file)
    lower_londa_index = closer(londa, lower_londa)
    upper_londa_index = closer(londa, upper_londa)
    corrected_mean_specs = np.loadtxt(corrected_mean_specs_file)
    corrected_mean_specs = corrected_mean_specs.T
    
    plt.figure()
    for i in range(totalbins):
        plt.plot(londa, corrected_mean_specs[i], label='Bin %d' % i)
    ax = plt.gca()
    ax.set_xlabel(r'Wavelength (nm)')
    ax.set_ylabel('Photoluminiscence (a.u.)')
    plt.ylim([0,1500])
    plt.xlim([0.99*londa[lower_londa_index],1.01*londa[upper_londa_index]])
    plt.legend(loc='upper left')
    ax.axvline(londa[lower_londa_index], ymin=0, ymax=1, linestyle='--', color='k')
    ax.axvline(londa[upper_londa_index], ymin=0, ymax=1, linestyle='--', color='k')
    aux_folder = manage_save_directory(common_path, 'antistokes_in_bins')
    figure_name = os.path.join(aux_folder,'all_bins_corrected_antistokes_%s.png' % NP)
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(save_folder, 'antistokes_in_bins')
    figure_name = os.path.join(aux_folder,'all_bins_corrected_antistokes_%s.png' % NP)
    plt.savefig(figure_name)
    plt.close()
    
    corrected_mean_pow = np.loadtxt(corrected_mean_pow_file)
    corrected_mean_irrad = np.loadtxt(corrected_mean_irrad_file)
    bin_penalization = np.loadtxt(bin_penalization_file)
    
    energy = lambda_to_energy(londa)
    
    a = lower_londa_index
    b = upper_londa_index
    
    instrumental_err = 0.05 # power meter uncertainty
    
    dof = b - a + 1 - 1 # número de datos (ver ajuste con energía) MENOS los parámetros del ajuste 
    frozen_distro = sta.chi2(dof)
    
    R2_matrix = np.zeros([totalbins, totalbins])
    chi_squared_matrix = np.zeros([totalbins, totalbins])
    p_value_matrix = np.zeros([totalbins, totalbins])
    T_matrix = np.zeros([totalbins, totalbins])
    err_T_matrix = np.zeros([totalbins, totalbins])
    A_fitted_list = []
    err_A_fitted_list = []

    reference_bin = list(range(1, totalbins-2))
    for index in reference_bin:
        
        if corrected_mean_pow[index] == 0:
            print('\nBin %d is empty.' % index)
            print('Skipping bin as reference.')
            continue
        
        if bin_penalization[index] == 1:
            print('\nBin %d is penalized.' % index)
            print('Skipping bin as reference.')
            continue
        
        ##################### ANTI-STOKES vs LAMBDA ######################
        ##################### ANTI-STOKES vs LAMBDA ######################
        ##################### ANTI-STOKES vs LAMBDA ######################
        plt.figure()
        x = londa[a:b]
        list_of_bins = list(range(1, totalbins))
        for i in list_of_bins:    
            A = corrected_mean_pow[i]/corrected_mean_pow[index]
            y = corrected_mean_specs[i][a:b]/corrected_mean_specs[index][a:b]
            plt.plot(x, y, '-', label='Bin %d/%d - A %.2f' % (i, index, A))
        plt.grid(True)
        plt.legend(loc='best')
        ax = plt.gca()
#        ax.set_ylim([0,10])
        ax.set_xlim([londa[a],londa[b]])
        ax.set_xlabel(r'Wavelength (nm)')
        ax.set_ylabel('Quotient')
        aux_folder = manage_save_directory(save_folder,'antistokes_quotient')
        figure_name = os.path.join(aux_folder,'quotient_vs_lambda_ref_%02d_%s.png' % (index, NP))
        plt.savefig(figure_name)
        plt.close()
#        plt.show()

        ##################### ANTI-STOKES FIT ######################
        ##################### ANTI-STOKES FIT ######################
        ##################### ANTI-STOKES FIT ######################
        
        print('\nFitting quotient Q using bin %d as reference...' % index)

        plt.figure()
        x = energy[a:b]
        list_of_bins_to_fit = list(range(index+2,totalbins))
        for i in list_of_bins_to_fit:
            y = corrected_mean_specs[i][a:b]/corrected_mean_specs[index][a:b]
            A = corrected_mean_pow[i]/corrected_mean_pow[index]
            err_A = instrumental_err * A * np.sqrt(2)
            plt.plot(x, y, '-', label='Bin %d/%d - A %.2f' % (i, index, A))
            if A == 0 or bin_penalization[i] == 1:
                T1 = 0
                T2 = 0
                err_T1 = 0
                err_T2 = 0
                fitted_A = 0
                err_A_fitted = 0
                r2_coef_pearson = 0
                chi_squared_pearson = 0
            else:
                init_params = [300, A]
                # Get the fitting parameters for the best quotient of photoluminiscence emission
                best_as, err = fit_quotient(init_params, x, A, err_A, y)
                # retrieve best fitted parameters
                fitted_A = best_as[1]
                T1 = best_as[0]
                T2 = fitted_A * (T1 - Tzero) + Tzero
                # calculate the errors and goodes of the fit
                err_A_fitted = np.sqrt(err[1,1])
                err_T1 = np.sqrt(err[0,0])
                cov_T1_A_fitted = err[0,1]
                err_T2 = np.sqrt( (fitted_A*err_T1)**2 + \
                                 ((T1 - Tzero)*err_A_fitted)**2 + \
                                 2*fitted_A*(T1 - Tzero)*cov_T1_A_fitted )
                yfit = quotient(x, *best_as)
                r2_coef_pearson = calc_r2(y, yfit)
                chi_squared_pearson = calc_chi_squared(y, yfit)
                plt.plot(x, yfit, '--k')
            # Asign matrix elements
            A_fitted_list.append(fitted_A)
            err_A_fitted_list.append(err_A_fitted)
            T_matrix[index,i] = T1
            T_matrix[i,index] = T2
            err_T_matrix[index,i] = err_T1
            err_T_matrix[i,index] = err_T2
            R2_matrix[index,i] = r2_coef_pearson
            R2_matrix[i,index] = r2_coef_pearson
            chi_squared_matrix[index,i] = chi_squared_pearson
            chi_squared_matrix[i,index] = chi_squared_pearson
            p_value_matrix[index,i] = 1 - frozen_distro.cdf(chi_squared_pearson)
            p_value_matrix[i,index] = 1 - frozen_distro.cdf(chi_squared_pearson)
            print('---------- Bin', i, 'Penalization:', bin_penalization[i], \
                  '\nR-sq: ', r2_coef_pearson, \
                  '\nA: ', A, 'error A:', err_A, \
                  '\nA_fitted: ', fitted_A, 'error A_fitted:', err_A_fitted, \
                  '\nT_1: ', T1, 'error T_1:', err_T1, \
                  '\nT_2: ', T2, 'error T_2:', err_T2)
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_xlim([energy[b],energy[a]])
#        ax.set_ylim([0.9,10])
        ax.set_xlabel('Energy (eV)',)
        ax.set_ylabel('Quotient')
        aux_folder = manage_save_directory(save_folder,'antistokes_quotient')
        figure_name = os.path.join(aux_folder,'quotient_vs_energy_ref_%02d_%s.png' % (index, NP))
        plt.savefig(figure_name)
        plt.close()
        
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    
    # Writing/creating files
    aux_folder = manage_save_directory(save_folder,'matrix')
    
    T_matrix_file = os.path.join(aux_folder,'Temp_matrix_%s.dat' % NP)
    
    err_T_matrix_file = os.path.join(aux_folder,'err_T_matrix_%s.dat' % NP)
    
    R2_matrix_file = os.path.join(aux_folder,'R2_matrix_%s.dat' % NP)
    
    p_value_matrix_file = os.path.join(aux_folder,'p_value_matrix_%s.dat' % NP)
    
    np.savetxt(T_matrix_file, T_matrix, delimiter=',', fmt='%.3e')
    
    np.savetxt(err_T_matrix_file, err_T_matrix, delimiter=',', fmt='%.3e')
    
    np.savetxt(R2_matrix_file, R2_matrix, delimiter=',', fmt='%.3e')
    
    np.savetxt(p_value_matrix_file, p_value_matrix, delimiter=',', fmt='%.3e')
    
    irradiance_file = os.path.join(aux_folder,'irradiance_matrix_%s.dat' % NP)
    
    np.savetxt(irradiance_file, corrected_mean_irrad, fmt='%.3e') 
    
    A_file = os.path.join(aux_folder,'A_%s.dat' % NP)
    
    np.savetxt(A_file, A_fitted_list, fmt='%.3e') 

    return


if __name__ == '__main__':
    
    # Parameters to load
    totalbins = 10 #total curve in the end
    meas_pow_bfp = 0.505 # in mW
    Tzero = 295 # in K
    window, deg, repetitions = 91, 0, 1
    factor = 0.47 # factor de potencia en la muestra
    image_size_px = 12 # IN PIXELS
    image_size_um = 0.8 # IN um
    camera_px_length = 1002 # number of pixels (number of points per spectrum)
    
    start_notch = 524 # in nm where notch starts ~525 nm (safe zone)
    end_notch = 543 # in nmwhere notch starts ~540 nm (safe zone)
    end_power = 590 # in nm from end_notch up to this lambda we measure power 580 nm
    start_spr = end_notch # in nm lambda from where to fit lorentz
    
    lower_londa = 510 # 503 nm
    upper_londa = start_notch
    
    plot_flag = False # if True will save all spectra's plots for each pixel
    
    base_folder = '/home/mariano/datos_mariano/posdoc/experimentos_PL_arg'
    NP_folder = 'AuNP_SS_80/20190905_repetitividad/201090905-144638_Luminescence 10x10 NP1'
    parent_folder = os.path.join(base_folder, NP_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders = [f for f in list_of_folders if re.search('Confocal_Spectrum',f)]
    list_of_folders.sort()
    
    path_to = os.path.join(parent_folder,'processed_data')
    
    for f in list_of_folders:
        folder = os.path.join(parent_folder,f)
        
        print('\n>>>>>>>>>>>>>>',f)
        
        process_confocal_to_bins(folder, path_to, totalbins, image_size_px, image_size_um, 
                                camera_px_length, window, deg, repetitions, 
                                factor, meas_pow_bfp, start_notch, end_notch,
                                end_power, start_spr, lower_londa, 
                                upper_londa, plot_flag=plot_flag)
        
        calculate_quotient(folder, path_to, totalbins, lower_londa, upper_londa, 
                           Tzero)
        
#        calculate_inverse_quotient(folder, path_to, totalbins, lower_londa, upper_londa, 
#                           Tzero)