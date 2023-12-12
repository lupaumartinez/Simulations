# -*- coding: utf-8 -*-
"""
Analysis of temperature increase of single AuNPs

Mariano Barella

21 aug 2018

CIBION, Buenos Aires, Argentina
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from functions_for_photoluminiscence import manage_save_directory, fit_linear

try:
    plt.style.use('for_confocal.mplstyle')
except:
    print('Pre-defined matplotlib style was not loaded.')

plt.ioff()
plt.close('all')

def calculate_temp_increase(folder, path_to, Tzero, totalbins, alpha, R2th):
    
    # Load folders
    NP =  folder.split('Spectrum_')[-1]
    
    save_folder = os.path.join(path_to, NP)
    
    common_path = os.path.join(path_to,'common_plots')

    folder = os.path.join(save_folder, 'matrix')
    
    # List files in folder and grab the ones we care
    list_of_files = os.listdir(folder)
    list_of_files.sort()
    
    T_matrix_file = [f for f in list_of_files if re.search('Temp_matrix',f)][0]
    T_matrix_file = os.path.join(folder, T_matrix_file)
    T_matrix = np.loadtxt(T_matrix_file, delimiter=',')
    T_matrix = T_matrix - Tzero
    
    err_T_matrix_file = [f for f in list_of_files if re.search('err_T_matrix',f)][0]
    err_T_matrix_file = os.path.join(folder, err_T_matrix_file)
    err_T_matrix = np.loadtxt(err_T_matrix_file, delimiter=',')
    
    R2_matrix_file = [f for f in list_of_files if re.search('R2_matrix',f)][0]
    R2_matrix_file = os.path.join(folder, R2_matrix_file)
    R2_matrix = np.loadtxt(R2_matrix_file, delimiter=',')
    R2_matrix = np.abs(R2_matrix)
    
    p_value_matrix_file = [f for f in list_of_files if re.search('p_value_matrix',f)][0]
    p_value_matrix_file = os.path.join(folder, p_value_matrix_file)
    p_value_matrix = np.loadtxt(p_value_matrix_file, delimiter=',')
    
    irradiance_matrix_file = [f for f in list_of_files if re.search('irradiance_matrix',f)][0]
    irradiance_matrix_file = os.path.join(folder, irradiance_matrix_file)
    mean_irrad = np.loadtxt(irradiance_matrix_file)
    
    print('\n-- NP ', NP)

    # ALLOCATION
    T_good_matrix = np.zeros([totalbins,totalbins])
    err_T_good_matrix = np.zeros([totalbins,totalbins])
    irrad_good = np.array([])
    T_avg = np.array([])
    T_err = np.array([])
    
    # APPLY CRITERIA TO ELIMINATE UNDESIRED/OUTLIERS/BAD-FITTED POINTS
    good_p_value = p_value_matrix > alpha
    good_r2 = R2_matrix > R2th
    good = good_r2 & good_p_value
    
    T_good_matrix[good] = T_matrix[good]
    err_T_good_matrix[good] = err_T_matrix[good]
    
    # Do some minor statistics
    for i in range(totalbins):
        N = np.sum(good[i,:])
        if not N: 
            print('Bin %d has no temperature values that fulfill our criteria.' % i)
            continue
        # temp list
        T_list = [aux for aux in T_good_matrix[i,:] if not aux == 0]
        T_list = np.array(T_list)
        M = len(T_list)
        # error list
        err_T_list = [aux for aux in err_T_good_matrix[i,:] if not aux == 0]
        err_T_list = np.array(err_T_list)
        # weights
        list_T_weights = (1/err_T_list)**2
        norm_weights = 1/np.sum(list_T_weights)
        list_T_weights = list_T_weights*norm_weights
        # append average of T_list (matrix's row)
        aux_avg = np.average(T_list, weights = list_T_weights)
        T_avg = np.append(T_avg, aux_avg)
        # append error of average of T_list (matrix's row)
        aux_err = (1/M)*np.sqrt(np.sum((err_T_list*list_T_weights)**2))
        T_err = np.append(T_err, aux_err)
        # append irradiance
        irrad_good = np.append(irrad_good, mean_irrad[i])
        print('Bin %d has %d temperature values.' % (i, M))
#        print('Temp. increase', T_list)
#        print('Errors', err_T_list)
#        print('Weights', list_T_weights)
       
   	# Fit slope for each NP in order to find information about medium dissipation
    x = irrad_good
    y = T_avg
    y_err = T_err
    
    # calculation of weights for weaighted least squares
    weights = (1/y_err)**2
    norm_weights = 1/np.sum(weights)
    weights = weights*norm_weights
    weights = []
	
    ############### FORCING NON ZERO INTERCEPT #####################
    ############### FORCING NON ZERO INTERCEPT #####################
    ############### FORCING NON ZERO INTERCEPT #####################

    print('NO forcing')
    slope_no_f, intercept_no_f, err_slope_no_f, err_c, r_squared_no, \
        x_fitted, y_fitted = fit_linear(x, y, weights, intercept=True)

    plt.figure()
    plt.errorbar(x, y, yerr = y_err, fmt = 'o', color = 'C0', label = NP,
                 ms = 5, mfc = 'C0', ecolor = 'C0', lw = 1, capsize = 3, 
                 barsabove = False)
    plt.plot(x_fitted, y_fitted, 'k--', label='Linear fit')
    ax = plt.gca()
    ax.set_xlabel(u'Irradiance (mW/µm$^{2}$)')
    ax.set_ylabel('Temperature increase (K)')
    ax.set_xlim([0, 2.5])
#    x_axis = list(np.arange(0,3.1,0.5))
#    ax.set_xticks(x_axis)
#    ax.set_xticklabels(x_axis)
#    ax.set_ylim([-10, 100])
    aux_folder = manage_save_directory(folder,'temp_vs_irrad_no_forcing')
    figure_name = os.path.join(aux_folder, 'no_forcing_temp_vs_irrad_R2th_%s_%s.png' % (str(R2th),NP))
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'temp_vs_irrad_no_forcing')
    figure_name = os.path.join(aux_folder, 'no_forcing_temp_vs_irrad_R2th_%s_%s.png' % (str(R2th),NP))
    plt.savefig(figure_name)
    plt.close()
    
    ############### FORCING ZERO INTERCEPT #####################
    ############### FORCING ZERO INTERCEPT #####################
    ############### FORCING ZERO INTERCEPT #####################
        
    print('YES forcing')
    slope_yes_f, intercept_yes_f, err_slope_yes_f, err_c, r_squared_yes, \
        x_fitted, y_fitted = fit_linear(x, y, weights, intercept=False)
    plt.figure()
    plt.errorbar(x, y, yerr = y_err, fmt = 'o', color = 'C0', label = NP,
                 ms = 5, mfc = 'C0', ecolor = 'C0', lw = 1, capsize = 3, 
                 barsabove = False)
    plt.plot(x_fitted, y_fitted, 'k--', label='Linear fit')
    ax = plt.gca()
    ax.set_xlabel(u'Irradiance (mW/µm$^{2}$)')
    ax.set_ylabel('Temperature increase (K)')
    ax.set_xlim([0, 2.5])
#    x_axis = list(np.arange(0,3.1,0.5))
#    ax.set_xticks(x_axis)
#    ax.set_xticklabels(x_axis)
#    ax.set_ylim([-10, 100])
    aux_folder = manage_save_directory(folder,'temp_vs_irrad_yes_forcing')
    figure_name = os.path.join(aux_folder, 'yes_forcing_temp_vs_irrad_R2th_%s_%s.png' % (str(R2th),NP))
    plt.savefig(figure_name)
    aux_folder = manage_save_directory(common_path,'temp_vs_irrad_yes_forcing')
    figure_name = os.path.join(aux_folder, 'yes_forcing_temp_vs_irrad_R2th_%s_%s.png' % (str(R2th),NP))
    plt.savefig(figure_name)
    plt.close()
    
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    
    # Writing/creating files
    aux_folder = manage_save_directory(folder,'temp_vs_irrad_no_forcing')
    slope_file = os.path.join(aux_folder, 'a_slope_no_forcing_R2th_%s_%s.dat' % (str(R2th),NP))
    f = open(slope_file, 'w+')
    f.write('SLOPE_(K*um2/mW) ERROR_SLOPE_(K*um2/mW) INTERCEPT_(K) R_SQUARED\n')
    string_to_write = '%.3e %.3e %.3e %.3e \n' % (slope_no_f, err_slope_no_f,\
                                                 intercept_no_f, r_squared_no)
    f.write(string_to_write)
    f.close()
    
    aux_folder = manage_save_directory(folder,'temp_vs_irrad_yes_forcing')
    slope_file = os.path.join(aux_folder, 'a_slope_yes_forcing_R2th_%s_%s.dat' % (str(R2th),NP))
    f = open(slope_file, 'w+')
    f.write('SLOPE_(K*um2/mW) ERROR_SLOPE_(K*um2/mW) R_SQUARED\n')
    string_to_write = '%.3e %.3e %.3e \n' % (slope_yes_f, err_slope_yes_f, \
                                            r_squared_yes)
    f.write(string_to_write)
    f.close()
    
    step2ok = True
    
    return step2ok
    
if __name__ == '__main__':
    
    # Parameters to load
    # Threhsold: check if data-point is going to be included in the analysis
    # If both criteria do not apply, erase datapoint
    alpha = 0.05 # alpha level for chi-squared test (compare with p-value), coarse criteria
    R2th = 0.6 # correlation coefficient threhsold, fine criteria
    Tzero = 295 # in K
    totalbins = 10
    
    base_folder = '/home/mariano/datos_mariano/posdoc/experimentos_PL_arg'
    NP_folder = 'AuNP_SS_80/20190905_repetitividad/201090905-144638_Luminescence 10x10 NP1'
    parent_folder = os.path.join(base_folder, NP_folder)
    list_of_folders = os.listdir(parent_folder)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
    list_of_folders = [f for f in list_of_folders if re.search('Confocal_Spectrum',f)]
    list_of_folders.sort()
    
    path_to = os.path.join(parent_folder,'processed_data')
    
    step2ok = False
    
    for f in list_of_folders:
        folder = os.path.join(parent_folder,f)
        print('\n>>>>>>>>>>>>>>',f)
        step2ok = calculate_temp_increase(folder, path_to, Tzero, totalbins, alpha, R2th)
    