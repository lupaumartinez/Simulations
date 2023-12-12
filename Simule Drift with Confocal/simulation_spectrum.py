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

def gaussian(x, Io, xo, wo):

    return Io*np.exp(-2*(x-xo)**2/(wo**2))

def fit_gaussian(gaussian, x, y):
    
    mean = sum(x * y)
    	#sigma = sum(y * (x - mean)**2)
    sigma = np.sqrt(sum(y*(x - mean)**2))

    popt, pcov = curve_fit(gaussian, x, y, p0 = [1, mean, sigma])
        
    #x_fit = np.linspace(x[0], x[-1], 100)
    pixel_size = (x[-1]-x[0])/100 
    x_fit = np.arange(x[0], x[-1], pixel_size)
    
    y_fit = gaussian(x_fit, popt[0], popt[1], popt[2])
    
    return [x_fit, y_fit], popt

def curve_gauss(image, rango):

	profile_x = np.mean(image, axis = 1)
	profile_y = np.mean(image, axis = 0)

	#axe = np.linspace(-rango/2, rango/2, image.shape[0])
	pixel_size = rango/image.shape[0] # in nm
	axe = np.arange(-rango/2, rango/2, pixel_size)

	return [axe, profile_x, profile_y]

def image_confocal_NP(wavelength, confocal_spectrum):
     
    desired_range = np.where((wavelength > 540) & (wavelength < 580))

    image = np.zeros((confocal_spectrum.shape[0], confocal_spectrum.shape[1]))
    
    for i in range(confocal_spectrum.shape[0]):
    	for j in range(confocal_spectrum.shape[1]):

    		spectrum_ij = confocal_spectrum[i, j, :]
    		intensity = np.mean(spectrum_ij[desired_range])

    		image[i, j] = intensity

    return image


def confocal_spectrum_NP(number_pixel, rango, xo, yo, Io, wavelength, londa_spr, starts_notch, ends_notch, To, K):

	A = np.zeros((number_pixel, number_pixel, len(wavelength)))
	#x = np.linspace(-rango/2, rango/2, number_pixel)

	pixel_size = rango/number_pixel # in nm
	x = np.arange(-rango/2, rango/2, pixel_size)

	for i in range(A.shape[0]):
		for j in range(A.shape[1]):

			r2 = (x[i]-xo)**2 + (x[j]-yo)**2
			Power = Io*np.exp(-2*r2/(0.320**2))
			wavelength, spectrum = spectrum_NP(Power, wavelength, londa_spr, starts_notch, ends_notch, To, K)
			A[i, j, :] = spectrum

	return wavelength, A

def spectrum_NP(Power, wavelength, londa_spr, starts_notch, ends_notch, To, K):

	last_index_AS = np.where(wavelength >= starts_notch)[0][0]
	wavelength_AS = wavelength[:last_index_AS]

	first_index_S = np.where(wavelength >= ends_notch)[0][0]
	wavelength_S = wavelength[first_index_S:]

	init_params = np.array([Power, 50, londa_spr, 0.0], dtype=np.double)

	Temp = To + K*Power
	spectrum_AS = lorentz2(wavelength_AS, *init_params)*bose_einstein(wavelength_AS, Temp)
	
	spectrum_S =  lorentz2(wavelength_S, *init_params)

	#print(len(wavelength), len(wavelength_AS), len(wavelength_S), len(spectrum_AS), len(wavelength_S))

	spectrum = np.zeros(len(wavelength))
	spectrum[:last_index_AS] = spectrum_AS
	spectrum[first_index_S:] = spectrum_S

	return wavelength, spectrum


def lorentz2(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def bose_einstein(londa, temp):
    # Bose-Einstein distribution
    # temp must be in Kelvin
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    k = 0.000086173 # Boltzmann's constant in eV/K
    londa_exc = 532 # in nm
    kT = k*temp
    inside = hc * (1/londa - 1/londa_exc) / kT
    out = 1 / ( np.exp( inside ) - 1 )
    return out

def matrix_time(exposure_time, number_pixel):

    x = number_pixel
    y = number_pixel

    matrix_pixel_time = (np.arange(x*y).reshape(x,y) + 1)*exposure_time
 
    return matrix_pixel_time.T

def multiple_images(number_pixel, rango, xo, yo, Io, wavelength, londa_spr, starts_notch, ends_notch, To, K
	, vx, vy, rate_I, exposure_time):

    N = number_pixel**2
    out = np.zeros((number_pixel, number_pixel, len(wavelength), N))
    matrixtime = matrix_time(exposure_time, number_pixel)

    first_position = np.zeros((number_pixel, number_pixel, len(wavelength)))
    last_position = np.zeros((number_pixel, number_pixel, len(wavelength)))

    for i in range(number_pixel):
        for j in range(number_pixel):
            t = matrixtime[j,i]
            xf = xo + t*vx
            yf = yo + t*vy
            If = Io + 0.10*Io*np.sin(rate_I*t*np.pi)
            wavelength, spectrum_ij = confocal_spectrum_NP(number_pixel, rango, xf, yf, If, wavelength, londa_spr, starts_notch, ends_notch, To, K)
            out[:,:,:,i*number_pixel+j] = spectrum_ij

            if t == exposure_time:
               first_position = out[:,:,:,i*number_pixel+j]

            elif t == exposure_time*number_pixel**2:
                last_position = out[:,:,:,i*number_pixel+j]

    return wavelength, out, first_position, last_position

def image_adquisition(multiple_image):

	number_pixel = multiple_image.shape[0]
	number_spectrum = multiple_image.shape[2]

	image = np.zeros((number_pixel,number_pixel, number_spectrum))
	for i in range(number_pixel):
	    for j in range(number_pixel):
	        x = i*number_pixel+j
	        image[j,i,:] = multiple_image[j,i,:,x]

	return image

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def optical_transmition(wavelength, intensity):

	final_intensity = intensity*((0.05/(wavelength[-1]-wavelength[0]))*(wavelength-wavelength[0]) + 0.05)

	return final_intensity


#wavelength, spectrum = spectrum_NP(Power = 10, wavelength = np.linspace(500, 600, 1002), londa_spr = 550, starts_notch = 520, ends_notch = 540, To = 298, K =1)
#plt.figure()
#plt.plot(wavelength, spectrum)
#plt.show()

parent_folder = 'C:/Users/Alumno/Dropbox/Simule Drift with Confocal'
parent_folder = os.path.normpath(parent_folder)
path_to = os.path.join(parent_folder,'Simulation_spectrum_on_confocal')

#save_folder = manage_save_directory(path_to,'Confocal_Spectrum_NP_004')


meas_pow_bfp = 1000 #uW
factor = 0.47
meas_pow_sample = factor*meas_pow_bfp
#w0 = 0.320

#irrad_calc = 2*meas_pow_sample/(np.pi*w0**2)

Io = meas_pow_sample

#wavelength, confocal_spectrum = confocal_spectrum_NP(number_pixel = 12, rango = 800, xo = 10, yo = 10, Io = irrad_calc, wavelength = np.linspace(500, 600, 1002), londa_spr = 550, starts_notch = 520, ends_notch = 540, To = 298, K =1)

#name = os.path.join(save_folder, 'wavelength.txt')
#np.savetxt(name, wavelength)

#plt.figure()

#for i in range(confocal_spectrum.shape[0]):
#	for j in range(confocal_spectrum.shape[1]):

	#	spectrum_ij = confocal_spectrum[i, j, :]

	#	name = os.path.join(save_folder, 'Spectrum_i%04d_j%04d.txt' % (i, j))

	#	np.savetxt(name, spectrum_ij)

	#	plt.plot(wavelength, spectrum_ij)

#plt.show()

rate_I_max = 1/12 #Hz

wavelength, position_NP, first_position_NP, last_position_NP = multiple_images(number_pixel = 12, rango = 0.800, xo = 0.010, yo = 0.010, Io = Io,
	wavelength = np.linspace(500, 600, 1002), londa_spr = 540, starts_notch = 520, ends_notch = 540, To = 298, K =1, 
	vx = 0, vy = 0, rate_I = rate_I_max, exposure_time = 0.8)

adquisition_NP = image_adquisition(position_NP)

save_folder = manage_save_directory(path_to,'Confocal_Spectrum_NP_006')
name = os.path.join(save_folder, 'wavelength.txt')
np.savetxt(name, wavelength)

factor_fotones = 2500

plt.figure()

for i in range(adquisition_NP.shape[0]):
	for j in range(adquisition_NP.shape[1]):

		spectrum_ij = factor_fotones*optical_transmition(wavelength, adquisition_NP[i, j, :])

		name = os.path.join(save_folder, 'Spectrum_i%04d_j%04d.txt' % (i, j))

		np.savetxt(name, spectrum_ij)

		plt.plot(wavelength, spectrum_ij)

plt.show()

plot_bool = True

if plot_bool:

	image = image_confocal_NP(wavelength, adquisition_NP)
	image_curve_gauss = curve_gauss(image, rango = 800)
	image_fitx, ajuste_x = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]))
	image_fity, ajuste_y = fit_gaussian(gaussian, image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]))

	print('Gauss ajuste:', 'fast', ajuste_x, 'slow', ajuste_y)
	   
	fig, (ax3_1, ax3_2) = plt.subplots(1, 2)

	ax3_1.imshow(image)
	ax3_1.set_title('Image adquisition')
	ax3_1.grid(False)

	ax3_2.plot(image_curve_gauss[0], image_curve_gauss[1]/max(image_curve_gauss[1]), 'ro',label = 'Fast')
	ax3_2.plot(image_fitx[0], image_fitx[1], 'r-', alpha=0.5)
	ax3_2.plot(image_curve_gauss[0], image_curve_gauss[2]/max(image_curve_gauss[2]), 'bo', label = 'Slow')
	ax3_2.plot(image_fity[0], image_fity[1], 'b-', alpha=0.5)
	ax3_2.set_ylim(0, 1.2)
	ax3_2.legend(loc = 'upper right')

	fig.set_tight_layout(True)

	#figure_name = os.path.join(save_folder, 'velocity_drift_vfast_%s_vslow_%s.png' % (v_x, v_y))
	#plt.savefig(figure_name)

	plt.show()
    

# Power Sensor BS

meas_pow_bfp = 1 #mW
pow_BS = round((meas_pow_bfp/0.6)*0.4, 3)
pixel_time = 0.8 + 0.1 #s
number_pixel = 12
total_time = pixel_time*number_pixel**2
rate_I = 1/12 #Hz
rate_air = 1/(10*60)
t = np.arange(0, total_time, 0.300)
If = pow_BS + 0.08*pow_BS*np.sin(rate_I*t*np.pi)*np.sin(rate_air*t*np.pi)

plt.figure()
plt.plot(t, If)
plt.xlabel('Confocal Time (s)')
plt.ylabel('Power BS (mW)')
plt.show()