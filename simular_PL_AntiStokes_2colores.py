# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:50:13 2023

@author: lupau
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:28:00 2023

@author: lupau
"""


# DETERMINACION DE PARAMETROS
import os
import matplotlib.pyplot as plt
import numpy as np

h = 4.135667516e-15  # in eV*s
c = 299792458  # in m/s


def closer(x, value):
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
    hc = 1239.84193  # Plank's constant times speed of light in eV*nm
    energy = hc/londa
    return energy


def energy_to_lambda(energy):
    # Energy in eV and wavelength in nm
    hc = 1239.84193  # Plank's constant times speed of light in eV*nm
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
    k = 0.000086173  # Boltzmann's constant in eV/K
    aux = (energy - El) / (k*Temp)
    y = 1/(np.exp(aux) - 1)
    return y


def spectrum2(long, scattering, absorption, laser, size_notch, To, I, K, r):

    desired_AS = np.where(long < laser-int(size_notch/2))
    desired_S = np.where(long > laser + int(size_notch/2))

    long_AS = long[desired_AS]
    long_S = long[desired_S]

    Ispr = I*(scattering*10**12)  # mW

    energyAS = lambda_to_energy(long_AS)

    betha = absorption[closer(long, laser)]/(4*np.pi*(r*10**-9)*K)  # K.m2/mW

    betha = betha*10**12  # K.um2/mW

    betha = round(betha, 2)

    Temp = To + betha*I

    print('laser', laser, 'ratio', r, 'betha', betha, 'I', I, 'Temp', Temp)

    El = lambda_to_energy(laser)  # excitation wavelength in eV
    BE = bose(energyAS, El, Temp)

    AS = Ispr[desired_AS]*BE

    S = Ispr[desired_S]

  #  plt.figure()
   # plt.plot(long, Ispr)
  #  plt.plot(long_AS, AS)
   # plt.plot(long_S, S)
   # plt.ylim(0, I)
  #  plt.show()
    PL = np.zeros(len(long))
    PL[desired_S] = S
    PL[desired_AS] = AS

    size = 100
    desired = np.where(long_AS[-1]-size < long_AS)
    IAS = np.sum(AS[desired])

    desired2 = np.where(long_S < long_S[0]+size)
    IS = np.sum(S[desired2])

    return IAS, IS, long, PL


def open_data(paren_folder, n, laser, size_notch, To, I, K):

    folder = []

    for files in os.listdir(parent_folder):
        if files.endswith("%1s" % n):
            folder = os.path.join(parent_folder, files)

    print(folder)

    list_of_files = os.listdir(folder)
    list_of_files.sort()
    file = list_of_files[3]

    print(file)

    name = os.path.join(folder, file)
    data = np.genfromtxt(name, delimiter=',')

    wavelength = data[1:, 0]
    abso = data[1:, 3]
    sca = data[1:, 2]
    extinction = data[1:, 1]

    r = int(file.split('.')[0])/2

    ratio = r

    wave = np.linspace(wavelength[0], wavelength[-1], 1000)
    scattering = np.interp(wave, wavelength, sca)
    absorption = np.interp(wave, wavelength, abso)
    exc = np.interp(wave, wavelength, extinction)

#   plt.plot(wave, absorption, 'b')
#   plt.plot(wave, scattering, 'r')

    IAS, IS, long_PL, PL = spectrum2(wave, scattering, absorption, laser, size_notch, To, I, K, r)

    IAS_list = IAS

    IS_list = IS

    return ratio, wave, scattering, absorption, PL


plt.close('all')

parent_folder = r'C:\Users\lupau\OneDrive\Documentos\Luciana Martinez\Programa_Python\Simule Spectrum Growth'
parent_folder = os.path.normpath(parent_folder)

n = 1.33

size_notch = 20
To = 295
# I = 0.5  #mW/um2
K = 0.6  # W/Km
K = K*1000  # mW

lasers = [532, 415]

Is_green = 0.1
r, wavelength, scattering, absorption, PLg = open_data(parent_folder, n, lasers[0], size_notch, To, Is_green, K)
Abs_green = absorption[closer(wavelength, lasers[0])]*Is_green
Ispr_green = Is_green*(scattering*10**12)

Is_red = [0.25, 0.5, 1, 2]

plt.figure()
plt.plot(wavelength, PLg, color = 'g', label='%s' % lasers[0])

for i in range(len(Is_red)):
    
    I = Is_red[i]

    r, wavelength, scattering, absorption, PLr = open_data(parent_folder, n, lasers[1], size_notch, To, I, K)
    
 #   plt.plot(wavelength, PL, color = 'r', label='%s' % 642)

    Abs = Abs_green + absorption[closer(wavelength, lasers[1])]*I
    
    It = Is_green + I
    
    if lasers[1]<lasers[0]:
        desired_AS0 =  np.where(wavelength < lasers[1] - int(size_notch/2))
        desired_AS = np.where((lasers[1]+int(size_notch/2) < wavelength) & (wavelength< lasers[0]-int(size_notch/2)))
        desired_S = np.where(wavelength > lasers[0] + int(size_notch/2))
    else:
        
        desired_AS0 =  np.where(wavelength < lasers[0] - int(size_notch/2))
        desired_AS = np.where((lasers[0]+int(size_notch/2) < wavelength) & (wavelength< lasers[1]-int(size_notch/2)))
        desired_S = np.where(wavelength > lasers[1] + int(size_notch/2))
        
    long_AS0 = wavelength[desired_AS0]
    long_AS = wavelength[desired_AS]
    long_S = wavelength[desired_S]
    
    Ispr = I*(scattering*10**12)
    
    bethaIt = Abs/(4*np.pi*(r*10**-9)*K)  # K.m2/mW
    bethaIt = bethaIt*10**12  # K.um2/mW
    bethaIt = round(bethaIt, 2)
    
    Temp = To + bethaIt
    
    print('Total Temp', Temp)
    
    energyAS = lambda_to_energy(long_AS)
    El = lambda_to_energy(lasers[0])  # excitation wavelength in eV
    BE = bose(energyAS, El, Temp)
    
    AS = Ispr_green[desired_AS]*BE
    S = Ispr_green[desired_S]
    
    energyAS0 = lambda_to_energy(long_AS0)
    E0 = lambda_to_energy(lasers[1])  # excitation wavelength in eV
    BE0 = bose(energyAS0, E0, Temp)
    
    AS0 = Ispr[desired_AS0]*BE0
    S0 = Ispr[desired_AS]
    
    S02 = Ispr[desired_S]
    
    PL = np.zeros(len(wavelength))
    PL[desired_AS0] = AS0
    PL[desired_AS] = AS + S0
    PL[desired_S] = S + S02

    plt.plot(wavelength, PL, '--', label='juntos')
    
    desired = np.where((wavelength > 545) & (wavelength <600))
    
    IS = np.sum(PL[desired])#-PLr[desired])
    ISg = np.sum(PLg[desired])
    
    print('I', I, ((IS/ISg)-1)*(Is_green))

    plt.show()

#plt.xlim(500,600)
plt.legend()