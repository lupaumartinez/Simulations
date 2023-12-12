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

from matrix_spiral import to_spiral


def gaussian(x, Io, xo, wo):

    return Io*np.exp(-2*(x-xo)**2/(wo**2))

def fit_gaussian(gaussian, x, y):
    
    mean = sum(x * y)
    	#sigma = sum(y * (x - mean)**2)
    sigma = np.sqrt(sum(y*(x - mean)**2))

    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma])
        
    x_fit = np.linspace(x[0], x[-1], 100)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2])
    
    return [x_fit, y_fit], popt


def image_NP(number_pixel, rango, xo, yo, Io):

	A = np.zeros((number_pixel, number_pixel))

	x = np.linspace(-rango/2, rango/2, number_pixel)

	for i in range(A.shape[0]):
		for j in range(A.shape[1]):

			r2 = (x[i]-xo)**2 + (x[j]-yo)**2
			A[i, j] = Io*np.exp(-2*r2/(320**2))

	return A

def matrix_time(exposure_time, number_pixel, confocal_type):

    x = number_pixel
    y = number_pixel

    matrix_pixel_time = (np.arange(x*y).reshape(x,y) + 1)*exposure_time

    if confocal_type == 'classic':
        matrix_pixel_time = matrix_pixel_time
    elif confocal_type == 'spiral':
        matrix_pixel_time = to_spiral(matrix_pixel_time, 'cw')
 
    return matrix_pixel_time.T

def multiple_images(number_pixel, rango, xo, yo, Io, vx, vy, rate_I, exposure_time, confocal_type):

    N = number_pixel**2
    out = np.zeros((number_pixel, number_pixel, N))
    matrixtime = matrix_time(exposure_time, number_pixel, confocal_type)

    first_position = np.zeros((number_pixel, number_pixel))
    last_position = np.zeros((number_pixel, number_pixel))

    for i in range(number_pixel):
        for j in range(number_pixel):
            t = matrixtime[j,i]
            xf = xo + t*vx
            yf = yo + t*vy
            If = Io + t*rate_I
            out[:,:,i*number_pixel+j] = image_NP(number_pixel, rango, xf, yf, If)

            if t == exposure_time:
               first_position = out[:,:,i*number_pixel+j]

            elif t == exposure_time*number_pixel**2:
                last_position = out[:,:,i*number_pixel+j]

    return out, first_position, last_position

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

def curve_diff(image, rango):

	axe = np.linspace(-rango/2, rango/2, image.shape[0])

	dif_x = np.diff(image)
	dif_y = np.diff(image.T)

	print(dif_x, dif_y)

	return [axe, dif_x, dif_y]

def image_map_drift(number_pixel, rango, xo, yo, Io, vx_max, vy_max, N, rate_I, exposure_time, confocal_type):

	vx = np.linspace(-vx_max, vx_max, N)
	vy = np.linspace(-vy_max, vy_max, N)

	stack_adquisition = np.zeros((number_pixel, number_pixel, len(vx),len(vy)))

	for i in range(len(vx)):
		for j in range(len(vy)):

			stack_image, first_position, last_position = multiple_images(number_pixel, rango, xo, yo, Io, vx[i], vy[j], rate_I, exposure_time, confocal_type)
			adquisition = image_adquisition(stack_image)
			stack_adquisition[ :, :, i, j] = adquisition

	return vx, vy, stack_adquisition

def image_map_intensity(number_pixel, rango, xo, yo, Io, vx, vy, rate_I_max, N_I, exposure_time, confocal_type):

	rate_I = np.linspace(-rate_I_max, rate_I_max, N_I)

	stack_adquisition = np.zeros((number_pixel, number_pixel, len(rate_I)))

	for i in range(len(rate_I)):

		stack_image, first_position, last_position = multiple_images(number_pixel, rango, xo, yo, Io, vx, vy, rate_I[i], exposure_time, confocal_type)
		adquisition = image_adquisition(stack_image)
		stack_adquisition[ :, :, i] = adquisition

	return rate_I, stack_adquisition

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

#%%
    
#parent_folder = '//Fileserver/na/Luciana Martinez/Programa_Python/Simule Drift with Confocal'
parent_folder = 'C:/Users/Alumno/Dropbox/Simule Drift with Confocal'
parent_folder = os.path.normpath(parent_folder)
path_to = os.path.join(parent_folder,'Simulation_drift_intensity_on_confocal')

#parameters:
rango = 800 #nm
number_pixel = 12
exposure_time = 0.8 #s
confocal_type = 'classic' # 'spiral'

rate_I = 0
#rate_I = -round(0.01/60, 5) # - 1% por minuto, oscilacion laser
#rate_I = round(0.05/60, 5) 
# 60 nm growth radio = 0.12 nm/seg => scattering = radio**4, I = scattering**6

I_max = 1 + rate_I*exposure_time*number_pixel**2

vx_drift = 0.5
vy_drift = 0.5

save_folder = manage_save_directory(path_to,'range_%s_numberpixel_%s_pixeltime_%s_rateI_%s_confocal_%s' % (rango, number_pixel, exposure_time, rate_I, confocal_type))

position_NP, first_position_NP, last_position_NP = multiple_images(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0, Io = 1,
                      vx = vx_drift, vy = vy_drift, rate_I = rate_I, exposure_time = exposure_time, confocal_type = confocal_type)

#irst_position_NP = position_NP[:,:,0]
#last_position_NP = position_NP[:,:,number_pixel*number_pixel-1]
adquisition_NP = image_adquisition(position_NP)

first_curve_gauss = curve_gauss(first_position_NP, rango = rango)
last_curve_gauss = curve_gauss(last_position_NP, rango = rango)
adquisition_curve_gauss = curve_gauss(adquisition_NP, rango = rango)

first_fitx, first_ajuste_x = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[1]/max(first_curve_gauss[1]))
first_fity, first_ajuste_y = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[2]/max(first_curve_gauss[2]))
 
last_fitx, last_ajuste_x = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[1]/max(last_curve_gauss[1]))
last_fity, last_ajuste_y = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[2]/max(last_curve_gauss[2]))

ad_fitx, ad_ajuste_x = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[1]/max(adquisition_curve_gauss[1]))
ad_fity, ad_ajuste_y = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[2]/max(adquisition_curve_gauss[2]))

first_wox = round(first_ajuste_x[2], 3)
first_woy = round(first_ajuste_y[2], 3)

last_wox = round(last_ajuste_x[2], 3)
last_woy = round(last_ajuste_y[2], 3)

last_xo = round(last_ajuste_x[1], 3)
last_yo = round(last_ajuste_y[1], 3)

ad_xo = round(ad_ajuste_x[1], 3)
ad_yo = round(ad_ajuste_y[1], 3)

plt.close()

fig, ((ax1_1, ax2_1, ax3_1), (ax1_2, ax2_2, ax3_2)) = plt.subplots(2, 3)

ax1_1.imshow(first_position_NP, vmin=0, vmax=I_max)
ax1_1.set_title('First position')
ax1_1.grid(False)

ax2_1.imshow(last_position_NP, vmin=0, vmax=I_max)
ax2_1.set_title('Last position')
ax2_1.grid(False)

ax3_1.imshow(adquisition_NP, vmin=0, vmax=I_max)
ax3_1.set_title('Image adquisition')
ax3_1.grid(False)

ax1_2.plot(first_curve_gauss[0], first_curve_gauss[1]/max(first_curve_gauss[1]), 'ro', label = 'Fast')
ax1_2.plot(first_fitx[0], first_fitx[1], 'r-', alpha=0.5)
ax1_2.plot(first_curve_gauss[0], first_curve_gauss[2]/max(first_curve_gauss[2]), 'bo', label = 'Slow')
ax1_2.plot(first_fity[0], first_fity[1], 'b-', alpha=0.5)
ax1_2.set_ylim(0, 1.2)
ax1_2.legend(loc = 'upper right')

ax2_2.plot(last_curve_gauss[0], last_curve_gauss[1]/max(last_curve_gauss[1]), 'ro', label = 'Fast')
ax2_2.plot(last_fitx[0], last_fitx[1], 'r-', alpha=0.5)
ax2_2.plot(last_curve_gauss[0], last_curve_gauss[2]/max(last_curve_gauss[2]), 'bo', label = 'Slow')
ax2_2.plot(last_fity[0], last_fity[1], 'b-', alpha=0.5)
ax2_2.set_ylim(0, 1.2)
ax2_2.legend(loc = 'upper right')

ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[1]/max(adquisition_curve_gauss[1]), 'ro', label = 'Fast')
ax3_2.plot(ad_fitx[0], ad_fitx[1], 'r-', alpha=0.5)
ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[2]/max(adquisition_curve_gauss[2]), 'bo', label = 'Slow')
ax3_2.plot(ad_fity[0], ad_fity[1], 'b-', alpha=0.5)
ax3_2.set_ylim(0, 1.2)
ax3_2.legend(loc = 'upper right')

fig.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'velocity_drift_max_%s_%s.png' % (vx_drift, vy_drift))
plt.savefig(figure_name)
plt.close()

#%%
Number_velocity = 5
vx, vy, stack_adquisition =  image_map_drift(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0, Io = 1, 
                                             vx_max = vx_drift, vy_max = vy_drift, rate_I = rate_I, N = Number_velocity, exposure_time = exposure_time,  confocal_type = confocal_type)
vo_module = []

wo_mean = []
wox_list = []
woy_list = []

xo_list = []
yo_list = []

intensity_maximum = []

plot_all = True

for i in range(Number_velocity):
    for j in range(Number_velocity):
            
        image = stack_adquisition[:, :, i, j]
        image_curve_gauss = curve_gauss(image, rango = rango)
        image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]))
        image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]))

        intensity_maximum.append(np.max(image))

        wox = round(ajuste_x[2],3)
        woy = round(ajuste_y[2],3)

        xo = round(ajuste_x[1],3)
        yo = round(ajuste_y[1],3)

        v_x = round(vx[i], 3)
        v_y = round(vy[j], 3)
        v = round(np.sqrt(v_x**2 + v_y**2), 3)
        vo_module.append(v)

        wox_list.append(wox)
        woy_list.append(woy)
        wo_mean.append((wox + woy)/2)

        xo_list.append(xo)
        yo_list.append(yo)   
        
        if plot_all:
        
            fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
            
            ax3_1.imshow(image, vmin=0, vmax=I_max)
            ax3_1.set_title('Image adquisition')
            ax3_1.grid(False)
            
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast')
            ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow')
            ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
            ax3_2.set_ylim(0, 1.2)
            ax3_2.legend(loc = 'upper right')
            
            fig.set_tight_layout(True)
            
            figure_name = os.path.join(save_folder, 'velocity_drift_vfast_%s_vslow_%s.png' % (v_x, v_y))
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

#%%

fig_v3, ax_v3 = plt.subplots(1, 1)

ax_v3.set_title('Gauss Fit')

ax_v3.plot(np.array(yo_list), np.array(xo_list), 'ko')

#ax_v3.axhspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)
#ax_v3.axvspan(1 - error_tolerance, 1 + error_tolerance, facecolor='#2ca02c', alpha=0.5)

ax_v3.set_ylabel('Xo Fast')
ax_v3.set_xlabel('Xo Slow')

fig_v3.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'center_gauss.png')
plt.savefig(figure_name, dpi = 400)
        
plt.close()

#%%

fig_I, ax_v1 = plt.subplots(1, 1)

ax_v1.plot(vo_module, intensity_maximum, 'ko', alpha=0.5)

ax_v1.set_xlabel('Velocity module [nm/s]')
ax_v1.set_ylabel('Intensity maximum')

fig_I.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'intensity_maximum_vs_velocity_module.png')
plt.savefig(figure_name, dpi = 400)
        
plt.close()



#%%

error_tolerance = 0.05 

vx = -0.5
vy = -0.5

save_folder = manage_save_directory(path_to,'range_%s_numberpixel_%s_pixeltime_%s_vfast_%s_vslow_%s_confocal_%s' % (rango, number_pixel, exposure_time, vx, vy, confocal_type))

rate_I_max = round(0.05/60, 5) 
Number_intensity = 5
rate_I_array, stack_adquisition =  image_map_intensity(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0, Io = 1, 
                                             vx = vx, vy = vy, rate_I_max = rate_I_max, N_I = Number_intensity, exposure_time = exposure_time,  confocal_type = confocal_type)
rateI = []

intensity_maximum = []

plot_all = True

for i in range(Number_intensity):
            
    image = stack_adquisition[:, :, i]
    image_curve_gauss = curve_gauss(image, rango = rango)
    image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]))
    image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]))

    intensity_maximum.append(np.max(image))

    rate_I_value = round(rate_I_array[i], 4)
    rateI.append(rate_I_value)
        
    if plot_all:
    
        fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
        
        ax3_1.imshow(image, vmin=0, vmax=I_max)
        ax3_1.set_title('Image adquisition')
        ax3_1.grid(False)
        
        ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast')
        ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
        ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow')
        ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
        ax3_2.set_ylim(0, 1.2)
        ax3_2.legend(loc = 'upper right')
    
        fig.set_tight_layout(True)
        
        figure_name = os.path.join(save_folder, 'rate_intensity_%s_vfast_%s_vslow_%s.png' % (rate_I_value, vx, vy))
        plt.savefig(figure_name)

        plt.close()


fig_I, ax_v1 = plt.subplots(1, 1)

ax_v1.plot(rateI, intensity_maximum, 'ko', alpha=0.5)

ax_v1.set_xlabel('Rate Intensity [1/s]')
ax_v1.set_ylabel('Intensity maximum')

ax_v1.axhspan(np.max(first_position_NP) - error_tolerance, np.max(first_position_NP) + error_tolerance, facecolor='#2ca02c', alpha=0.5)
ax_v1.set_ylim(0.8, 1.2)

fig_I.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'intensity_maximum_vs_rate_intensity_vfast_%s_vslow_%s.png' % (vx, vy))
plt.savefig(figure_name, dpi = 400)
        
plt.close()
