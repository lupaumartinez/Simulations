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


def image_NP(number_pixel, rango, xo, yo):

	A = np.zeros((number_pixel, number_pixel))

	x = np.linspace(-rango/2, rango/2, number_pixel)

	for i in range(A.shape[0]):
		for j in range(A.shape[1]):

			r2 = (x[i]-xo)**2 + (x[j]-yo)**2
			A[i, j] = np.exp(-2*r2/(320**2))

	return A

def matrix_time(exposure_time, number_pixel, confocal_type):

    x = number_pixel
    y = number_pixel

    matrix_pixel_time = (np.arange(x*y).reshape(x,y) + 1)*exposure_time

    if confocal_type == 'spiral':
        matrix_pixel_time = to_spiral(matrix_pixel_time, 'cw')
    elif confocal_type == 'classical':
        matrix_pixel_time = matrix_pixel_time

    return matrix_pixel_time.T

def multiple_images(number_pixel, rango, xo, yo, t_blinking_initial, duration_blinking, exposure_time, confocal_type):

    N = number_pixel**2
    out = np.zeros((number_pixel, number_pixel, N))
    matrixtime = matrix_time(exposure_time, number_pixel, confocal_type)

    first_position = np.zeros((number_pixel, number_pixel))
    last_position = np.zeros((number_pixel, number_pixel))
     
    for i in range(number_pixel):
        for j in range(number_pixel):
            t = matrixtime[j,i]

            if t > t_blinking_initial and t < duration_blinking + t_blinking_initial:

                out[:,:,i*number_pixel+j] = np.zeros((number_pixel, number_pixel))

            else:

                out[:,:,i*number_pixel+j] = image_NP(number_pixel, rango, xo, yo)

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


def image_map_blinking(number_pixel, rango, xo, yo, t_blinking_initial, duration_blinking, N, exposure_time, confocal_type):

	t_blinking_initial_array = np.linspace(exposure_time, t_blinking_initial, N)
	duration_blinking_array = np.linspace(exposure_time, duration_blinking, N)

	stack_adquisition = np.zeros((number_pixel, number_pixel, len(t_blinking_initial_array),len(duration_blinking_array)))

	for i in range(len(t_blinking_initial_array)):
		for j in range(len(duration_blinking_array)):

			stack_image, first_position, last_position = multiple_images(number_pixel, rango, xo, yo, t_blinking_initial_array[i], duration_blinking_array[j], exposure_time, confocal_type)
			adquisition = image_adquisition(stack_image)
			stack_adquisition[ :, :, i, j] = adquisition

	return t_blinking_initial_array, duration_blinking_array, stack_adquisition

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
path_to = os.path.join(parent_folder,'Simulation_blinking_on_confocal')

#parameters:
rango = 800 #nm
number_pixel = 20
exposure_time = 1 #s
confocal_type = 'classical' #'classic'

t_blinking_initial = exposure_time*number_pixel*(number_pixel-1)
duration_blinking = exposure_time*number_pixel

save_folder = manage_save_directory(path_to,'range_%s_number_pixel_%s_pixel_time_%s_confocal_%s' % (rango, number_pixel, exposure_time, confocal_type))

position_NP, first_position_NP, last_position_NP = multiple_images(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0,
                      t_blinking_initial = t_blinking_initial, duration_blinking = duration_blinking, exposure_time = exposure_time, confocal_type = confocal_type)

#irst_position_NP = position_NP[:,:,0]
#last_position_NP = position_NP[:,:,number_pixel*number_pixel-1]
adquisition_NP = image_adquisition(position_NP)

first_curve_gauss = curve_gauss(first_position_NP, rango = rango)
last_curve_gauss = curve_gauss(last_position_NP, rango = rango)
adquisition_curve_gauss = curve_gauss(adquisition_NP, rango = rango)

first_fitx, first_ajuste_x = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[1])
first_fity, first_ajuste_y = fit_gaussian(gaussian, first_curve_gauss[0], first_curve_gauss[2])
 
last_fitx, last_ajuste_x = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[1])
last_fity, last_ajuste_y = fit_gaussian(gaussian, last_curve_gauss[0], last_curve_gauss[2])

ad_fitx, ad_ajuste_x = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[1])
ad_fity, ad_ajuste_y = fit_gaussian(gaussian, adquisition_curve_gauss[0], adquisition_curve_gauss[2])

first_wox = round(first_ajuste_x[2], 3)
first_woy = round(first_ajuste_y[2], 3)

last_wox = round(last_ajuste_x[2], 3)
last_woy = round(last_ajuste_y[2], 3)

last_xo = round(last_ajuste_x[1], 3)
last_yo = round(last_ajuste_y[1], 3)

ad_xo = round(ad_ajuste_x[1], 3)
ad_yo = round(ad_ajuste_y[1], 3)

print('Last xo', last_xo, 'Adquisition xo', ad_xo)
print('Last yo', last_yo, 'Adquisition yo', ad_yo)

plt.close()

fig, ((ax1_1, ax2_1, ax3_1), (ax1_2, ax2_2, ax3_2)) = plt.subplots(2, 3)

ax1_1.imshow(first_position_NP)
ax1_1.set_title('First position')
ax1_1.grid(False)

ax2_1.imshow(last_position_NP)
ax2_1.set_title('Last position')
ax2_1.grid(False)

ax3_1.imshow(adquisition_NP)
ax3_1.set_title('Image adquisition')
ax3_1.grid(False)

ax1_2.plot(first_curve_gauss[0], first_curve_gauss[1], 'ro', label = 'x')
ax1_2.plot(first_fitx[0], first_fitx[1], 'r-', alpha=0.5)
ax1_2.plot(first_curve_gauss[0], first_curve_gauss[2], 'bo', label = 'y')
ax1_2.plot(first_fity[0], first_fity[1], 'b-', alpha=0.5)
ax1_2.set_ylim(0, 0.6)
ax1_2.legend()

ax2_2.plot(last_curve_gauss[0], last_curve_gauss[1], 'ro', label = 'x')
ax2_2.plot(last_fitx[0], last_fitx[1], 'r-', alpha=0.5)
ax2_2.plot(last_curve_gauss[0], last_curve_gauss[2], 'bo', label = 'y')
ax2_2.plot(last_fity[0], last_fity[1], 'b-', alpha=0.5)
ax2_2.set_ylim(0, 0.6)
ax2_2.legend()

ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[1], 'ro', label = 'x')
ax3_2.plot(ad_fitx[0], ad_fitx[1], 'r-', alpha=0.5)
ax3_2.plot(adquisition_curve_gauss[0], adquisition_curve_gauss[2], 'bo', label = 'y')
ax3_2.plot(ad_fity[0], ad_fity[1], 'b-', alpha=0.5)
ax3_2.set_ylim(0, 0.6)
ax3_2.legend()

fig.set_tight_layout(True)

figure_name = os.path.join(save_folder, 'duration_blinking_%s.png' % (duration_blinking))
plt.savefig(figure_name)
plt.close()

#%%
Number = 4
t_blinking_initial_array, duration_blinking_array, stack_adquisition =  image_map_blinking(number_pixel = number_pixel, rango = rango, xo = 0, yo = 0, 
                                             t_blinking_initial = t_blinking_initial, duration_blinking = duration_blinking,
                                              N = Number, exposure_time = exposure_time,  confocal_type = confocal_type)

plot_all = True

for i in range(Number):
    for j in range(Number):
            
        image = stack_adquisition[:, :, i, j]
        image_curve_gauss = curve_gauss(image, rango = rango)
        image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[1])
        image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[2])

        duration_blinking = duration_blinking_array[j]
        t_blinking_initial = t_blinking_initial_array[i]
        
        if plot_all:
        
            fig, (ax3_1, ax3_2) = plt.subplots(1, 2)
            
            ax3_1.imshow(image)
            ax3_1.set_title('Image adquisition')
            ax3_1.grid(False)
            
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1], 'ro',label = 'x')
            ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
            ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2], 'bo', label = 'y')
            ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
            ax3_2.set_ylim(0, 0.6)
            ax3_2.legend()
            
            fig.set_tight_layout(True)
            
            figure_name = os.path.join(save_folder, 'initial_%s_duration_blinking_%s.png' % (t_blinking_initial, duration_blinking))
            plt.savefig(figure_name)
        
            plt.close()
        