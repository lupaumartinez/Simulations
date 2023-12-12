# -*- coding: utf-8 -*-
"""
Analysis of single AuNPs photoluminiscence spectra for temperature calculation 
acquired with PySpectrum at CIBION

Mariano Barella

16 aug 2019

based on "witec_data_photoluminiscense.py"

"""

import os
import re
import step1_process_data_photoluminiscense as step1
import step2_data_temp_increase as step2
import step3_data_slope_statistics_ask_forcing as step3
import step4_simulation_post_run_steps as step4

# Parameters to load
meas_pow_bfp = 0.50 # in mW
Tzero = 295 # in K
window, deg, repetitions = 3, 0, 1
factor = 0.47 # factor de potencia en la muestra
image_size_px = 16 # IN PIXELS
image_size_um = 0.8 # IN um
camera_px_length = 1002 # number of pixels (number of points per spectrum)

start_notch = 524 # in nm where notch starts ~525 nm (safe zone)
end_notch = 543 # in nmwhere notch starts ~540 nm (safe zone)
end_power = 590 # in nm from end_notch up to this lambda we measure power 580 nm
start_spr = end_notch # in nm lambda from where to fit lorentz
plot_flag = False # if True will save all spectra's plots for each pixel

#start_notch = 520 # in nm where notch starts ~525 nm (safe zone)
#end_notch = 540 # in nmwhere notch starts ~540 nm (safe zone)
#end_power = 590 # in nm from end_notch up to this lambda we measure power 580 nm
#start_spr = end_notch # in nm lambda from where to fit lorentz
#plot_flag = False # if True will save all spectra's plots for each pixel

lower_londa = 510 # 503 nm
upper_londa = start_notch

# Parameters to load
# Threhsold: check if data-point is going to be included i  n the analysis
# If both criteria do not apply, erase datapoint
alpha = 0.05 # alpha level for chi-squared test (compare with p-value), coarse criteria
R2th = 0.7 # correlation coefficient threhsold, fine criteria

threshold = 0.5 # threshold for R-squared of linear fit (temp increase vs irrad)

single_NP_flag = False # if True only one NP will be analyzed
NP_int = 11 # NP to be analyzed

step2ok = True

base_folder = 'C:/Users/Luciana/Dropbox/Simule Drift with Confocal Spectrum/'
NP_folder = 'number_pixel_16_pixel_time_0.8'

parent_folder = os.path.join(base_folder, NP_folder)
parent_folder = os.path.normpath(parent_folder)

list_of_folders = os.listdir(parent_folder)
list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(parent_folder,f))]
list_of_folders = [f for f in list_of_folders if re.search('Confocal_Spectrum',f)]
list_of_folders.sort()

totalbins_list = [7, 8, 9, 10] #number of bins

for bins in totalbins_list:

    totalbins = bins #number of bins

    path_to = os.path.join(parent_folder,'processed_data_bin_%02d'%(totalbins))

    if single_NP_flag:
        print('\nAnalyzing only NP %d.' % NP_int)
            
    for f in list_of_folders:
        if single_NP_flag:    
            if not re.search('_NP_%03d' % NP_int, f):
                continue
            else:
                print('\nTarget NP found!')
            
        folder = os.path.join(parent_folder,f)

        print('\n>>>>>>>>>>>>>>',f)

        print('\n===========STEP 1')

        step1.process_confocal_to_bins(folder, path_to, totalbins, image_size_px, image_size_um, 
                                camera_px_length, window, deg, repetitions, 
                                factor, meas_pow_bfp, start_notch, end_notch,
                                end_power, start_spr, lower_londa, 
                                upper_londa, plot_flag=plot_flag)
        
        step1.calculate_quotient(folder, path_to, totalbins, lower_londa, upper_londa, 
                           Tzero)
        
        print('\n===========STEP 2')

        step2ok = step2.calculate_temp_increase(folder, path_to, Tzero, totalbins, alpha, R2th)

    print('\n===========STEP 3')

    if step2ok and not single_NP_flag:
        forcing = 'no'
        step3.gather_data(path_to, forcing, R2th, totalbins, meas_pow_bfp)
        forcing = 'yes'
        step3.gather_data(path_to, forcing, R2th, totalbins, meas_pow_bfp)
        
        forcing = 'no'
        step3.statistics(path_to, forcing, R2th, totalbins, threshold)
        forcing = 'yes'
        step3.statistics(path_to, forcing, R2th, totalbins, threshold)
    else:
        print('\nSTEP 2 was not executed. Skipping STEP 3.')

    print('\nProcess done until STEP 3.')

    print('\n===========STEP 4')

if step2ok and not single_NP_flag:
    step4.plots_processed_data_bins(parent_folder)
else:
    print('\nSTEP 2 was not executed. Skipping STEP 3 and 4.')
print('\nProcess done.')











    
    
    
    
    
    
    