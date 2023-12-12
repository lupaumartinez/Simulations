# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:14:24 2019

@author: Luciana Martinez

CIBION, Bs As, Argentina

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


def gaussian(x, Io, xo, wo):

    return Io*np.exp(-2*(x-xo)**2/(wo**2))

def fit_gaussian(gaussian, x, y):
    
    mean = sum(x * y)
    	#sigma = sum(y * (x - mean)**2)
    sigma = np.sqrt(sum(y*(x - mean)**2)) 
    
    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma])
        
    x_fit = np.linspace(x[0], x[-1], 100)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2])
    
    return [x_fit, y_fit], round(popt[2],2)


def image_NP(number_pixel, rango, xo, yo):

	A = np.zeros((number_pixel, number_pixel))

	x = np.linspace(-rango/2, rango/2, number_pixel)

	for i in range(A.shape[0]):
		for j in range(A.shape[1]):

			r2 = (x[i]-xo)**2 + (x[j]-yo)**2
			A[i, j] = np.exp(-2*r2/(320**2))

	return A

def matrix_time(exposure_time, number_pixel):

    x = number_pixel
    y = number_pixel
    matrix_pixel_time = (np.arange(x*y).reshape(x,y) + 1)*exposure_time

    return matrix_pixel_time.T

def multiple_images(number_pixel, rango, xo, yo, vx, vy, exposure_time):

    N = number_pixel**2
    out = np.zeros((number_pixel, number_pixel, N))
    matrixtime = matrix_time(exposure_time, number_pixel)

    for i in range(number_pixel):
        for j in range(number_pixel):
            t = matrixtime[j,i]
            xf = xo + t*vx
            yf = yo + t*vy
            out[:,:,i*number_pixel+j] = image_NP(number_pixel, rango, xf, yf)

    return out

def image_adquisition(multiple_image):

	number_pixel = multiple_image.shape[0]

	image = np.zeros((number_pixel,number_pixel))
	for i in range(number_pixel):
	    for j in range(number_pixel):
	        x = i*number_pixel+j
	        image[j,i] = multiple_image[j,i,x]

	return image

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	axe = np.linspace(-rango/2, rango/2, image.shape[0])

	#axe = np.arange(0,image.shape[0]) + 1

	return [axe, profile_x, profile_y]


def image_map_drift(number_pixel, rango, xo, yo, vx_max, vy_max, N, exposure_time):

	vx = np.linspace(-vx_max, vx_max, N)
	vy = np.linspace(-vy_max, vy_max, N)

	stack_adquisition = np.zeros((number_pixel, number_pixel, len(vx),len(vy)))

	for i in range(len(vx)):
		for j in range(len(vy)):

			stack_image = multiple_images(number_pixel, rango, xo, yo, vx[i], vy[j], exposure_time)
			adquisition = image_adquisition(stack_image)
			stack_adquisition[ :, :, i, j] = adquisition

	return vx, vy, stack_adquisition

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

#%%
    
parent_folder = '//Fileserver/na/Luciana Martinez/Programa_Python/Simule Drift with Confocal'
parent_folder = os.path.normpath(parent_folder)
path_to = os.path.join(parent_folder,'Simulation_drift_on_confocal')

#parameters:
rango = 800 #nm
number_pixel = 12
exposure_time = 0.8 #s

vx_drift = 0.5
vy_drift = 0.5

save_folder = manage_save_directory(path_to,'range_%s_number_pixel_%s_pixel_time_%s' % (rango, number_pixel, exposure_time))

position_NP = multiple_images(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0,
                      vx = vx_drift, vy = vy_drift, exposure_time = exposure_time)

first_position_NP = position_NP[:,:,0]
last_position_NP = position_NP[:,:,number_pixel*number_pixel-1]
adquisition_NP = image_adquisition(position_NP)

first_curve_gauss = curve_gauss(first_position_NP, rango = rango)
last_curve_gauss = curve_gauss(last_position_NP, rango = rango)
adquisition_curve_gauss = curve_gauss(adquisition_NP, rango = rango)

first_fitx, first_wox = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[1])
first_fity, first_woy = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[2])
 
last_fitx, last_wox = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[1])
last_fity, last_woy = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[2])

ad_fitx, ad_wox = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[1])
ad_fity, ad_woy = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[2])


print('Gauss Fit: Wx', first_wox, last_wox, ad_wox)
print('Gauss Fit: Wy', first_woy, last_woy, ad_woy)

plt.close()

fig, ((ax1_1, ax2_1, ax3_1), (ax1_2, ax2_2, ax3_2)) = plt.subplots(2, 3)

ax1_1.imshow(first_position_NP)
ax1_1.set_title('First position')
ax1_1.grid(False)

ax2_1.imshow(last_position_NP)
ax2_1.set_title('Last position')
ax2_1.grid(False)

ax3_1.imshow(adquisition_NP)
ax3_1.set_title('Image acquisition')
ax3_1.grid(False)

ax1_2.plot(first_curve_gauss[0], first_curve_gauss[1], 'ro', label = 'Fast Axis')
ax1_2.plot(first_fitx[0], first_fitx[1], 'r-', alpha=0.5)
ax1_2.plot(first_curve_gauss[0], first_curve_gauss[2], 'bo', label = 'Slow Axis')
ax1_2.plot(first_fity[0], first_fity[1], 'b-', alpha=0.5)
ax1_2.set_ylim(0, 0.6)
ax1_2.legend()

ax2_2.plot(last_curve_gauss[0], last_curve_gauss[1], 'ro', label = 'Fast Axis')
ax2_2.plot(last_fitx[0], last_fitx[1], 'r-', alpha=0.5)
ax2_2.plot(last_curve_gauss[0], last_curve_gauss[2], 'bo', label = 'Slow Axis')
ax2_2.plot(last_fity[0], last_fity[1], 'b-', alpha=0.5)
ax2_2.set_ylim(0, 0.6)
ax2_2.legend()

ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[1], 'ro', label = 'Fast Axis')
ax3_2.plot(ad_fitx[0], ad_fitx[1], 'r-', alpha=0.5)
ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[2], 'bo', label = 'Slow Axis')
ax3_2.plot(ad_fity[0], ad_fity[1], 'b-', alpha=0.5)
ax3_2.set_ylim(0, 0.6)
ax3_2.legend()

fig.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'velocity_drift_max_%s_%s.png' % (vx_drift, vy_drift))
plt.savefig(figure_name)
plt.close()

#%%
Number_velocity = 5
vx, vy, stack_adquisition =  image_map_drift(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0, 
                                             vx_max = vx_drift, vy_max = vy_drift, N = Number_velocity, exposure_time = exposure_time)

vo_module = []

wo_mean = []
wox_list = []
woy_list = []

plot_all = True

for i in range(Number_velocity):
    for j in range(Number_velocity):
            
        image = stack_adquisition[:, :, i, j]
        image_curve_gauss = curve_gauss(image, rango = rango)
        image_fitx, wox = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[1])
        image_fity, woy = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[2])
        
        v_x = vx[i]
        v_y = vy[j]
        v = np.sqrt(v_x**2 + v_y**2)
        vo_module.append(v)
        
        wox_list.append(wox)
        woy_list.append(woy)
        wo_mean.append((wox + woy)/2)   
        
        print('Wx', wox, 'Wy', woy, 'vx' , v_x, 'vy', v_y)
        
        if plot_all:
        
            fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
            
            ax3_1.imshow(image)
            ax3_1.set_title('Image acquisition')
            ax3_1.grid(False)
            
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1], 'ro',label = 'Fast Axis')
            ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2], 'bo', label = 'Slow axis')
            ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
            ax3_2.set_ylim(0, 0.6)
            ax3_2.legend()
            
            fig.set_tight_layout(True)
            
            figure_name = os.path.join(save_folder, 'velocity_drift_%s_%s.png' % (v_x, v_y))
            plt.savefig(figure_name)
        
            plt.close()
        
#%%
            
error_tolerance = 0.05 # %5 de tolerancia en que la confocal erre al omega del ajuste gaussiano           
            
#%%       
fig_v, ax_v = plt.subplots(1, 1)

ax_v.set_title('Gauss Fit')

ax_v.plot(np.array(woy_list)/320, np.array(wox_list)/320, 'ko')

ax_v.axhspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_v.axvspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)

ax_v.set_ylabel('Ratio Wo Fast')
ax_v.set_xlabel('Ratio Wo Slow')

fig_v.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'ratio_wfast_vs_ratio_wslow.png')
plt.savefig(figure_name)
        
plt.close()


#%%

fig_v2, (ax_v1, ax_v2, ax_v3) = plt.subplots(1, 3)

ax_v1.plot(vo_module, np.array(wox_list)/320, 'ro', alpha=0.5)
ax_v2.plot(vo_module, np.array(woy_list)/320, 'bo', alpha=0.5)
ax_v3.plot(vo_module, np.array(wo_mean)/320, 'mo', alpha=0.5)

ax_v1.set_xlabel('Velocity module [nm/s]')
ax_v2.set_xlabel('Velocity module [nm/s]')
ax_v3.set_xlabel('Velocity module [nm/s]')
ax_v1.set_ylabel('Ratio Wo Fast')
ax_v2.set_ylabel('Ratio Wo Slow')
ax_v3.set_ylabel('Ratio Wo mean')

ax_v1.axhspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_v2.axhspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_v3.axhspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
             
ax_v1.set_ylim(0.80, 1.2)
ax_v2.set_ylim(0.80, 1.2)
ax_v3.set_ylim(0.80, 1.2)

fig_v2.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'ratio_w_vs_velocity_module.png')
plt.savefig(figure_name, dpi = 400)
        
plt.close()


#%%

fig_hist, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3)

ax_h1.hist(np.array(wox_list)/320, bins=10, range=[0.80, 1.2], normed = True, rwidth=0.9, color='r', alpha=0.5)
ax_h2.hist(np.array(woy_list)/320, bins=10, range=[0.80, 1.2], normed = True, rwidth=0.9, color='b', alpha=0.5)
ax_h3.hist(np.array(wo_mean)/320, bins=10, range=[0.80, 1.2], normed = True, rwidth=0.9, color='m', alpha=0.5)

ax_h1.set_xlabel('Ratio Wo Fast')
ax_h2.set_xlabel('Ratio Wo Slow')
ax_h3.set_xlabel('Ratio Wo mean')
ax_h1.set_ylabel('Frequency')

ax_h1.axvspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_h2.axvspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_h3.axvspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)

ax_h1.set_ylim(0, 20)
ax_h2.set_ylim(0, 20)
ax_h3.set_ylim(0, 20)

fig_hist.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'histogram_ratio_w.png')
plt.savefig(figure_name, dpi = 400)
        
plt.close()

#%% Cuenta

size_scan = 800 #nm
number_pixel_array = np.arange(8,21)
pixel_size_array = size_scan/number_pixel_array
velocity_drift_max = 0.5 #nm/s
exposure_time_array = np.array([0.5, 0.8, 1, 2]) #s
#drift_max_per_pixel = (velocity_drift/pixel_size_array)/pixel_size_array#*exposure_time_array

time_total = number_pixel_array**2#*exposure_time_array
drift_max_per_pixel = velocity_drift_max*time_total/pixel_size_array

plt.figure()
plt.axhspan(0.5, 1, facecolor='0.5', alpha=0.5)
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[0], '--', label = 't_{exp}_{px} = 0.5 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[1], '--', label = 't_{exp}_{px} = 0.8 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[2], '--', label = 't_{exp}_{px} = 1.0 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[3], '--', label = 't_{exp}_{px} = 2.0 s')
plt.legend()
plt.xlabel('Number of pixels')
plt.ylabel('Drift max / pixel size')
plt.title('Size scan = 800 nm, Vel. Drift = 0.5 nm/s')
plt.ylim(0, 3)
plt.xlim(7.5, 20.5)

#plt.savefig(parent_folder, 'confocal_limitations_by_drift.png')
plt.close()

