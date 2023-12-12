# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 17:07:46 2019

@author: Cibion2
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt


def velocity_drift(path_from):
                       
    filename = 'velocity.txt' 
    filepath = os.path.join(path_from, filename)
                       
    data = np.loadtxt(filepath, skiprows=1)
   # vx = data[1]
   # vy = data[2]
    vmodule = data[:,3]

    return vmodule
    
def info(path_from):
                       
    filename = 'Info.txt' 
    filepath = os.path.join(path_from, filename)
                       
    data = np.loadtxt(filepath, dtype = 'str')
    k = float(data[6,1]) #[K/mW/um2]

    return k

def stats_data(path_from, forcing):

    list_of_folders = os.listdir(path_from)
    list_of_folders = [f for f in list_of_folders if os.path.isdir(os.path.join(path_from,f))]

    list_of_folders_bin = [f for f in list_of_folders if re.search('processed_data_bin_',f)]
    list_of_folders_bin.sort()
    L = len(list_of_folders_bin)

    list_of_NP = [f for f in list_of_folders if re.search('NP',f)]
    NP = len(list_of_NP)
    
    list_of_slopes = np.zeros((NP, L))
    list_of_slopes_err = np.zeros((NP, L))
    list_of_intercept = np.zeros((NP, L))
    list_of_r_sq_slope = np.zeros((NP, L))

    list_of_wx = np.zeros((NP, L))
    list_of_wx_err = np.zeros((NP, L))
    list_of_wy = np.zeros((NP, L))
    list_of_wy_err = np.zeros((NP, L))
    
    for i in range(L):
        bineado = list_of_folders_bin[i]
        slope_folder = os.path.join(path_from, bineado, 'stats')
    
        # import temp increase data    
        filename = 'all_NP_data_%s_forcing.dat' % (forcing)
        slope_filepath = os.path.join(slope_folder, filename)
        data = np.loadtxt(slope_filepath, skiprows=1)
        
        if forcing == 'yes':
            list_of_slopes[:, i] = data[:,0]
            list_of_slopes_err[:, i] = data[:,1]
            list_of_r_sq_slope[:, i] = data[:,2]
            list_of_wx[:, i] = data[:,7]
            list_of_wx_err[:, i] = data[:,8]
            list_of_wy[:, i] = data[:,10]
            list_of_wy_err[:, i] = data[:,11]
        elif forcing == 'no':
            list_of_slopes[:, i] = data[:,0]
            list_of_slopes_err[:, i] = data[:,1]
            list_of_intercept[:, i] = data[:,2]
            list_of_r_sq_slope[:, i] = data[:,3]
            list_of_wx[:, i] = data[:,7]
            list_of_wx_err[:, i] = data[:,8]
            list_of_wy[:, i] = data[:,10]
            list_of_wy_err[:, i] = data[:,11]
        else:
            raise ValueError('forcing value can only be yes or no.')
        
    return list_of_slopes, list_of_slopes_err, list_of_wx, list_of_wy, list_of_folders_bin

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def plots_processed_data_bins(parent_folder):

    save_folder = manage_save_directory(parent_folder ,'simulation_figures_processed_data_bin')

    k = info(parent_folder)
    vmodule = velocity_drift(parent_folder)

    list_of_slopes_no, list_of_slopes_err_no, list_of_wx_no, list_of_wy_no, list_of_folders_bin = stats_data(parent_folder, 'no')
    list_of_slopes_yes, list_of_slopes_err_yes, list_of_wx_yes, list_of_wy_yes, list_of_folders_bin = stats_data(parent_folder, 'yes')

    L_bins = list_of_slopes_no.shape[1]
    print(list_of_folders_bin)
    slopes_yes = np.zeros((L_bins))
    slopes_no = np.zeros((L_bins))
    err_slopes_yes = np.zeros((L_bins))
    err_slopes_no = np.zeros((L_bins))
    total_bin = np.zeros((L_bins))

    wx_yes = np.zeros((L_bins))
    wy_yes = np.zeros((L_bins))
    wx_no = np.zeros((L_bins))
    wy_no = np.zeros((L_bins))

    #L_NP = list_of_slopes_no.shape[0]
    #NP = np.arange(0, L_NP, 1) + 1
    #index_NP = np.where(vmodule <= 0.000001)[0][0] 
    #name_NP = NP[index_NP]

    name_NP = 5
    index_NP = name_NP - 1

    for i in range(L_bins):

        total_bin[i] = int(list_of_folders_bin[i].split('_')[3])

        plt.figure()
        plt.title(list_of_folders_bin[i])
        slopes_bin_yes = list_of_slopes_yes[:,i]
        slopes_bin_no = list_of_slopes_no[:,i]

        err_yes = list_of_slopes_err_yes[:,i]
        err_no = list_of_slopes_err_no[:,i]

        #plt.plot(NP, slopes_bin, 'o')
        plt.errorbar(vmodule, slopes_bin_yes, yerr= err_yes , fmt = 'o', label = 'yes forcing')
        plt.errorbar(vmodule, slopes_bin_no,  yerr= err_no, fmt = 'o', label = 'no forcing')

        plt.axhline(k, linestyle = '--', color = 'grey')
        plt.ylabel('Slopes T vs I [K/mW/um2]')
        plt.xlabel('Drift velocity module [um/s]')
        plt.ylim(k*0.8, k*1.2)
        plt.legend()

        figure_name = os.path.join(save_folder, 'slopes_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        wx_bin_yes = list_of_wx_yes[:,i]
        wx_bin_no = list_of_wx_no[:,i]

        wy_bin_yes = list_of_wy_yes[:,i]
        wy_bin_no = list_of_wy_no[:,i]

        plt.plot(vmodule, wx_bin_yes, 'o', label = 'wx yes forcing')
        plt.plot(vmodule, wx_bin_no,  'o', label = 'wx no forcing')
        plt.plot(vmodule, wy_bin_yes, 'o', label = 'wy yes forcing')
        plt.plot(vmodule, wy_bin_no,  'o', label = 'wy no forcing')

        plt.axhline(0.320, linestyle = '--', color = 'grey')
        plt.ylabel('W')
        plt.xlabel('Drift velocity module [um/s]')
        plt.legend()

        figure_name = os.path.join(save_folder, 'w_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        plt.figure()
        plt.title(list_of_folders_bin[i])
        plt.hist(slopes_bin_yes, bins=5, range=[k*0.8, k*1.2], rwidth = 1, \
             align='mid', alpha = 0.6, normed = False, label = 'yes forcing')

        plt.hist(slopes_bin_no, bins=5, range=[k*0.8, k*1.2], rwidth = 1, \
             align='mid', alpha = 0.6, normed = False, label = 'no forcing')

        plt.axvline(k, linestyle = '--', color = 'grey')
        plt.ylabel('Frequency')
        plt.xlabel('Slopes T vs I [K/mW/um2]')
        plt.xlim(k*0.8, k*1.2)
        plt.ylim(0, 5)
        plt.legend()

        figure_name = os.path.join(save_folder, 'hist_slopes_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        slopes_yes[i] = slopes_bin_yes[index_NP]
        slopes_no[i] = slopes_bin_no[index_NP]
        err_slopes_yes[i] = err_yes[index_NP]
        err_slopes_no[i] = err_no[index_NP]

        wx_no[i] = wx_bin_no[index_NP]
        wy_no[i] = wy_bin_no[index_NP]
        wx_yes[i] = wx_bin_yes[index_NP]
        wy_yes[i] = wy_bin_yes[index_NP]

    plt.figure()
    plt.title('NP_%03d'%name_NP)
    plt.errorbar(total_bin, slopes_yes, yerr= err_slopes_yes, fmt = 'o', label = 'yes forcing')
    plt.errorbar(total_bin, slopes_no, yerr= err_slopes_no, fmt = 'o', label = 'no forcing')
    plt.axhline(k, linestyle = '--', color = 'grey')
    plt.ylabel('Slopes T vs I [K/mW/um2]')
    plt.xlabel('Number of total bins')
    plt.ylim(k*0.8, k*1.2)
    plt.legend()

    figure_name = os.path.join(save_folder, 'NP_%03d_slopes_all_bin.png'%name_NP)
    plt.savefig(figure_name, dpi = 400)
        
    plt.close()
    

    plt.figure()
    plt.title('NP_%03d'%name_NP)

    plt.plot(total_bin, wx_yes, 'o', label = 'wx yes forcing')
    plt.plot(total_bin, wx_no, 'o', label = 'wx no forcing')

    plt.plot(total_bin, wy_yes, 'o', label = 'wy yes forcing')
    plt.plot(total_bin, wy_no, 'o', label = 'wy no forcing')

    plt.axhline(0.320, linestyle = '--', color = 'grey')
    plt.ylabel('W [um]')
    plt.xlabel('Number of total bins')
    #plt.ylim(0.320*0.8, 0.320*1.2)
    plt.legend()

    figure_name = os.path.join(save_folder, 'NP_%03d_w_all_bin.png'%name_NP)
    plt.savefig(figure_name, dpi = 400)
        
    plt.close()

    return
    
def real_plots_processed_data_bins(parent_folder, name_NP):

    save_folder = manage_save_directory(parent_folder ,'figures_processed_data_bin')

    list_of_slopes_no, list_of_slopes_err_no, list_of_wx_no, list_of_wy_no, list_of_folders_bin = stats_data(parent_folder, 'no')
    list_of_slopes_yes, list_of_slopes_err_yes, list_of_wx_yes, list_of_wy_yes, list_of_folders_bin = stats_data(parent_folder, 'yes')

    L_bins = list_of_slopes_no.shape[1]
    print(list_of_folders_bin)
    slopes_yes = np.zeros((L_bins))
    slopes_no = np.zeros((L_bins))
    err_slopes_yes = np.zeros((L_bins))
    err_slopes_no = np.zeros((L_bins))
    total_bin = np.zeros((L_bins))

    wx_yes = np.zeros((L_bins))
    wy_yes = np.zeros((L_bins))
    wx_no = np.zeros((L_bins))
    wy_no = np.zeros((L_bins))

    L_NP = list_of_slopes_no.shape[0]
    NP = np.arange(0, L_NP, 1) + 1

    index_NP = name_NP - 1

    for i in range(L_bins):

        total_bin[i] = int(list_of_folders_bin[i].split('_')[3])

        plt.figure()
        plt.title(list_of_folders_bin[i])
        slopes_bin_yes = list_of_slopes_yes[:,i]
        slopes_bin_no = list_of_slopes_no[:,i]

        err_yes = list_of_slopes_err_yes[:,i]
        err_no = list_of_slopes_err_no[:,i]

        plt.errorbar(NP, slopes_bin_yes, yerr= err_yes , fmt = 'o', label = 'yes forcing')
        plt.errorbar(NP, slopes_bin_no,  yerr= err_no, fmt = 'o', label = 'no forcing')

        #plt.axhline(k, linestyle = '--', color = 'grey')
        plt.ylabel('Slopes T vs I [K/mW/um2]')
        plt.xlabel('NP')
        plt.ylim(0, 150)
        plt.legend()

        figure_name = os.path.join(save_folder, 'slopes_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        wx_bin_yes = list_of_wx_yes[:,i]
        wx_bin_no = list_of_wx_no[:,i]

        wy_bin_yes = list_of_wy_yes[:,i]
        wy_bin_no = list_of_wy_no[:,i]

        plt.plot(NP, wx_bin_yes, 'o', label = 'wx yes forcing')
        plt.plot(NP, wx_bin_no,  'o', label = 'wx no forcing')
        plt.plot(NP, wy_bin_yes, 'o', label = 'wy yes forcing')
        plt.plot(NP, wy_bin_no,  'o', label = 'wy no forcing')

        #plt.axhline(0.320, linestyle = '--', color = 'grey')
        plt.ylabel('W')
        plt.xlabel('NP')
        plt.legend()

        figure_name = os.path.join(save_folder, 'w_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        plt.figure()
        plt.title(list_of_folders_bin[i])
        plt.hist(slopes_bin_yes, bins=15, range=[0, 150], rwidth = 1, \
             align='mid', alpha = 0.6, normed = False, label = 'yes forcing')

        plt.hist(slopes_bin_no, bins=15, range=[0, 150], rwidth = 1, \
             align='mid', alpha = 0.6, normed = False, label = 'no forcing')

        #plt.axvline(k, linestyle = '--', color = 'grey')
        plt.ylabel('Frequency')
        plt.xlabel('Slopes T vs I [K/mW/um2]')
        plt.xlim(0, 150)
        plt.ylim(0, 5)
        plt.legend()

        figure_name = os.path.join(save_folder, 'hist_slopes_%1s.png' % list_of_folders_bin[i])
        plt.savefig(figure_name, dpi = 400)
            
        plt.close()

        slopes_yes[i] = slopes_bin_yes[index_NP]
        slopes_no[i] = slopes_bin_no[index_NP]
        err_slopes_yes[i] = err_yes[index_NP]
        err_slopes_no[i] = err_no[index_NP]

        wx_no[i] = wx_bin_no[index_NP]
        wy_no[i] = wy_bin_no[index_NP]
        wx_yes[i] = wx_bin_yes[index_NP]
        wy_yes[i] = wy_bin_yes[index_NP]

    plt.figure()
    plt.title('NP_%03d'%name_NP)
    plt.errorbar(total_bin, slopes_yes, yerr= err_slopes_yes, fmt = 'o', label = 'yes forcing')
    plt.errorbar(total_bin, slopes_no, yerr= err_slopes_no, fmt = 'o', label = 'no forcing')
   # plt.axhline(k, linestyle = '--', color = 'grey')
    plt.ylabel('Slopes T vs I [K/mW/um2]')
    plt.xlabel('Number of total bins')
    plt.ylim(0, 150)
    plt.legend()

    figure_name = os.path.join(save_folder, 'NP_%03d_slopes_all_bin.png'%name_NP)
    plt.savefig(figure_name, dpi = 400)
        
    plt.close()
    

    plt.figure()
    plt.title('NP_%03d'%name_NP)

    plt.plot(total_bin, wx_yes, 'o', label = 'wx yes forcing')
    plt.plot(total_bin, wx_no, 'o', label = 'wx no forcing')

    plt.plot(total_bin, wy_yes, 'o', label = 'wy yes forcing')
    plt.plot(total_bin, wy_no, 'o', label = 'wy no forcing')

    #plt.axhline(0.320, linestyle = '--', color = 'grey')
    plt.ylabel('W [um]')
    plt.xlabel('Number of total bins')
    #plt.ylim(0.320*0.8, 0.320*1.2)
    plt.legend()

    figure_name = os.path.join(save_folder, 'NP_%03d_w_all_bin.png'%name_NP)
    plt.savefig(figure_name, dpi = 400)
        
    plt.close()

    return

if __name__ == '__main__':
    
    #parent_folder = '//fileserver/NA/Luciana Martinez/Programa_Python/Simule Drift with Confocal Spectrum/number_pixel_16_pixel_time_0.3'
    parent_folder = 'C:/Users/Alumno/Dropbox/Simule Drift with Confocal Spectrum/number_pixel_10_pixel_time_0.8'
    parent_folder = os.path.normpath(parent_folder)

    #plots_processed_data_bins(parent_folder) #para simulaciones
    real_plots_processed_data_bins(parent_folder, name_NP = 5)