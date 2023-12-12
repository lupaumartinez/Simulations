# -*- coding: utf-8 -*-
"""
Analysis of temperature increase of single AuNPs

Mariano Barella

21 aug 2019

CIBION, Buenos Aires, Argentina
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from functions_for_photoluminiscence import manage_save_directory

try:
    plt.style.use('for_confocal.mplstyle')
except:
    print('Pre-defined matplotlib style was not loaded.')

plt.ioff()
plt.close('all')

def gather_data(path_from, forcing, R2th, totalbins, power_bfp):

    #################### SLOPE STATISTICS ############################
    #################### SLOPE STATISTICS ############################
    #################### SLOPE STATISTICS ############################
    
    list_of_folders = os.listdir(path_from)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(path_from,f))]
    list_of_folders = [f for f in list_of_folders if re.search('NP',f)]
    list_of_folders.sort()
    L = len(list_of_folders)
    
    list_of_slopes = np.zeros((L))
    list_of_slopes_err = np.zeros((L))
    list_of_intercept = np.zeros((L))
    list_of_r_sq_slope = np.zeros((L))
    list_of_londa_max = np.zeros((L))
    list_of_width_pl = np.zeros((L))
    list_of_r_sq_spr = np.zeros((L))
    list_of_w0x = np.zeros((L))
    list_of_err_w0x = np.zeros((L))
    list_of_r_sq_w0x = np.zeros((L))
    list_of_w0y = np.zeros((L))
    list_of_err_w0y = np.zeros((L))
    list_of_r_sq_w0y = np.zeros((L))
    list_of_col = np.zeros((L))
    list_of_nps = np.zeros((L))
    
    for i in range(L):
        NP = list_of_folders[i]
        slope_folder = os.path.join(path_from, NP, 'matrix', 'temp_vs_irrad_%s_forcing' % forcing)
    
        # import temp increase data    
        filename = 'a_slope_%s_forcing_R2th_%s_%s.dat' % (forcing, str(R2th), NP)
        slope_filepath = os.path.join(slope_folder, filename)
        data = np.loadtxt(slope_filepath, skiprows=1)
        if forcing == 'yes':
            list_of_slopes[i] = data[0]
            list_of_slopes_err[i] = data[1]
            list_of_r_sq_slope[i] = data[2]
        elif forcing == 'no':
            list_of_slopes[i] = data[0]
            list_of_slopes_err[i] = data[1]
            list_of_intercept[i] = data[2]
            list_of_r_sq_slope[i] = data[3]
        else:
            raise ValueError('forcing value can only be yes or no.')
    
        # import spr data    
        spr_folder = os.path.join(path_from, NP, 'spr')
        filename = 'spr_fitted_parameters_%s.dat' % NP
        spr_filepath = os.path.join(spr_folder, filename)
        data = np.loadtxt(spr_filepath)
        list_of_londa_max[i] = data[0]
        list_of_width_pl[i] = data[1]
        list_of_r_sq_spr[i] = data[2]
        
        # import spr data    
        psf_folder = os.path.join(path_from, NP, 'psf')
        filename = 'psf_fitted_parameters_%s.dat' % NP
        spr_filepath = os.path.join(psf_folder, filename)
        data = np.loadtxt(spr_filepath)
        list_of_w0x[i] = data[0]
        list_of_err_w0x[i] = data[1]
        list_of_r_sq_w0x[i] = data[2]
        list_of_w0y[i] = data[3]
        list_of_err_w0y[i] = data[4]
        list_of_r_sq_w0y[i] = data[5]
        
        list_of_col[i] = NP.split('_')[1]
        list_of_nps[i] = NP.split('_')[3]
        
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    #################### SAVE DATA ############################
    
    aux_folder = manage_save_directory(path_from,'stats')

    to_save = np.array([list_of_slopes, 
                         list_of_slopes_err, 
                         list_of_intercept, 
                         list_of_r_sq_slope,
                         list_of_londa_max, 
                         list_of_width_pl, 
                         list_of_r_sq_spr,
                         list_of_w0x,
                         list_of_err_w0x,
                         list_of_r_sq_w0x,
                         list_of_w0y,
                         list_of_err_w0y,
                         list_of_r_sq_w0y,
                         list_of_col,
                         list_of_nps]).T
    header_text = 'SLOPE_mW_um2_K ERROR_SLOPE_mW_um2_K INTERCEPT_K SLOPE_R_SQ LAMBDA_MAX_nm WIDTH_nm SPR_R_SQ WAIST_X_um ERROR_WAIST_X_um WAIST_X_R_SQ WAIST_Y_um ERROR_WAIST_Y_um WAIST_Y_R_SQ COL NP'
    
    path_to_save = os.path.join(aux_folder,'all_NP_data_%s_forcing.dat' % forcing)
    for s in range(totalbins):
        np.savetxt(path_to_save, to_save, fmt='%.3e', header=header_text, comments='')
        
    return

def statistics(path_from, forcing, R2th, totalbins, threshold):
    
    stats_file = os.path.join(path_from,'stats','all_NP_data_%s_forcing.dat' % forcing)
    data = np.loadtxt(stats_file, skiprows=1)

    list_of_slopes = data[:,0]
    list_of_slopes_err = data[:,1]
    list_of_intercept = data[:,2]
    list_of_r_sq_slope = data[:,3]
    list_of_londa_max = data[:,4]
    list_of_width_pl = data[:,5]
    list_of_r_sq_spr = data[:,6]  
    list_of_w0x = data[:,7]
    list_of_err_w0x = data[:,8]
    list_of_r_sq_w0x = data[:,9]
    list_of_w0y = data[:,10]
    list_of_err_w0y = data[:,11]
    list_of_r_sq_w0y = data[:,12]
    list_of_col = data[:,13]   
    list_of_nps = data[:,14]
    
    include = np.where(list_of_r_sq_slope > threshold)
    list_of_londa_max_included = list_of_londa_max[include]
    list_of_slopes_included = list_of_slopes[include]
    list_of_col_included = list_of_col[include]
    
    plt.figure()
#    plt.plot(list_of_londa_max, list_of_slopes, 'o', alpha = 0.6)
#    plt.plot(list_of_londa_max_included, list_of_slopes_included, 'o', alpha = 0.6)
    plt.plot(list_of_col, list_of_slopes, 'o', alpha = 0.6)
    plt.plot(list_of_col_included, list_of_slopes_included, 'o', alpha = 0.6)
    ax = plt.gca()
#    ax.set_xlim([530, 570])
    ax.set_ylim([-5, 100])
#    ax.set_xlabel(r'Max PL wavelength (nm)')
    ax.set_xlabel(r'NP')
    ax.set_ylabel(r'Temp. increase per irrad. (K µm$^{2}$/mW)')
    aux_folder = manage_save_directory(path_from,'stats')
    figure_name = os.path.join(aux_folder,'a_slope_vs_lambda_%s_forcing_R2th_%s_%s_bines.png' % \
                               (forcing, str(R2th), str(totalbins)))
    plt.savefig(figure_name)
    plt.close()
    
    # do a little bit of stats
    mu = np.mean(list_of_slopes_included)
    std = np.std(list_of_slopes_included, ddof=1)
    
    borders = [mu-std, mu+std]
    
#    from itertools import groupby
#        
#    print([list(j) for i, j in zip(list_of_slopes, groupby(list_of_col))])
    
    plt.figure()
#    plt.hist(list_of_slopes, bins=15, range=[0,150], rwidth = 1, \
#             align='mid', alpha = 0.6, edgecolor='k', normed = False)
#    plt.hist(list_of_slopes_included, bins=15, range=[0,150], rwidth = 1, \
#             align='mid', color='k', alpha = 0.4, edgecolor='k', normed = False)
    plt.hist(list_of_slopes, bins=15, range=[0,150], rwidth = 1, \
             align='mid', alpha = 0.6, edgecolor='k', normed = False)
    plt.hist(list_of_slopes_included, bins=15, range=[0,150], rwidth = 1, \
             align='mid', color='k', alpha = 0.4, edgecolor='k', normed = False)
    plt.legend(loc=1, handlelength=0)
    ax = plt.gca()
#    ax.text(10,0.06,power_bfp,fontsize=8,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    ax.axvline(mu, ymin = 0, ymax = 1, color='k', linestyle='--', linewidth=2)
    ax.fill_between([borders[0], borders[1]], [0,0], [100,100],  facecolor='k', alpha=0.25)
    ax.get_yaxis().set_ticks([])
    ax.set_ylim([0, 25])
    ax.set_xlim([-10, 160])
    ax.set_xlabel(u'Temp. increase per irrad. (µm$^{2}$K/mW)')
    #ax.set_ylabel('Counts', fontsize=16)
    #ax.set_xlim([0, 2])
    aux_folder = manage_save_directory(path_from,'stats')
    figure_name = os.path.join(aux_folder,'hist_slopes_%s_forcing_R2th_%s_%s_bines.png' % \
                               (forcing, str(R2th), str(totalbins)))
    plt.savefig(figure_name)
    plt.close()

    return

if __name__ == '__main__':
    
    totalbins = 10 # total curve in the end
    power_bfp = 0.505

    # Parameters to load
    # Threhsold: check if data-point is going to be included in the analysis
    # If both criteria do not apply, erase datapoint
    alpha = 0.05 # alpha level for chi-squared test (compare with p-value), coarse criteria
    R2th = 0.6 # correlation coefficient threhsold, fine criteria
    
    threshold = 0.5 # threshold for R-squared of linear fit (temp increase vs irrad)
    
    base_folder = '/home/mariano/datos_mariano/posdoc/experimentos_PL_arg'
    NP_folder = 'AuNP_SS_80/20190905_repetitividad/201090905-144638_Luminescence 10x10 NP1'
    parent_folder = os.path.join(base_folder, NP_folder)
    
    path_from = os.path.join(parent_folder,'processed_data')

    gather_data(path_from, 'no', R2th, totalbins, power_bfp)
    gather_data(path_from, 'yes', R2th, totalbins, power_bfp)
    
    statistics(path_from, 'no', R2th, totalbins, threshold)
    statistics(path_from, 'yes', R2th, totalbins, threshold)
        
