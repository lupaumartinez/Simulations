# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:28:00 2023

@author: lupau
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

## DETERMINACION DE PARAMETROS
h = 4.135667516e-15 # in eV*s
c = 299792458 # in m/s

def closer(x,value):
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
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
    energy = hc/londa
    return energy

def energy_to_lambda(energy):
    # Energy in eV and wavelength in nm
    hc = 1239.84193 # Plank's constant times speed of light in eV*nm
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
    k = 0.000086173 # Boltzmann's constant in eV/K
    aux = (energy - El) / (k*Temp)
    y = 1/( np.exp(aux) - 1 )
    return y

def spectrum(long, laser, size_notch, To, betha532, I, longSPR, gammaSPR):
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    desired_S = np.where(long > laser+ int(size_notch/2)  )
    
    long_AS = long[desired_AS]
    long_S =  long[desired_S]
    
    p =  longSPR, gammaSPR
    
    Ispr = I*lorentz(long, *p)
    
    energyAS = lambda_to_energy(long_AS)
    
    betha = int(betha532*(Ispr[closer(long, laser)]/Ispr[closer(long, 532)]))
    
    print('laser', laser, 'betha', betha)
    
    Temp = To + betha*I
    El = lambda_to_energy(laser) # excitation wavelength in eV
    print('laser', laser, 'E eV', El)
    BE = bose(energyAS, El, Temp) 
    
    AS = Ispr[desired_AS]*BE
    
    S = Ispr[desired_S]
    
  #  plt.figure()
   # plt.plot(long, Ispr)
  #  plt.plot(long_AS, AS)
   # plt.plot(long_S, S)
   # plt.ylim(0, I)
  #  plt.show()
    
    size = 2000
    desired = np.where(long_AS[-1]-size<long_AS)
    IAS = np.sum(AS[desired])
    
    desired2 = np.where(long_S<long_S[0]+size)
    IS = np.sum(S[desired2])
    
    r = IAS/IS
    
    return Ispr, IAS, IS, r

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out

def lorentz2(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz(p, x, y):
    return curve_fit(lorentz2, x, y, p0 = p)

def calc_r2(observed, fitted):
    # Calculate coefficient of determination
    avg_y = observed.mean()
    ssres = ((observed - fitted)**2).sum()
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

def fit_spr(spectrum, wavelength, starts, ends):

    desired_range_stokes = np.where((wavelength > starts) & (wavelength < ends))
    wavelength_S = wavelength[desired_range_stokes]
    spectrum_S = spectrum[desired_range_stokes]
    spectrum_S = spectrum_S/max(spectrum_S)

    init_londa = (starts + ends)/2  

    init_params2 = np.array([1, 80, init_londa, 0.05], dtype=np.double)
    best_lorentz, err = fit_lorentz(init_params2, wavelength_S, spectrum_S)

    lorentz_fitted = lorentz2(wavelength_S, *best_lorentz)
    r2_coef_pearson = calc_r2(spectrum_S, lorentz_fitted)

    full_lorentz_fitted = lorentz(wavelength, *best_lorentz)
    londa_max_pl = best_lorentz[2]
    width_pl = best_lorentz[1]

    return full_lorentz_fitted, londa_max_pl, width_pl

def spectrum2(long, scattering, absorption, laser, size_notch, To, T, K, r):
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    desired_S = np.where(long > laser+ int(size_notch/2) )
    
    long_AS = long[desired_AS]
    long_S =  long[desired_S]
    
    energyAS = lambda_to_energy(long_AS)
    
    betha = absorption[closer(long, laser)]/(4*np.pi*(r*10**-9)*K)# K.m2/mW
    
    betha = betha*10**12 # K.um2/mW
    
    betha = round(betha, 2)
   
    Temp = T #300 #K To + betha*I
    
    I = (Temp - To)/betha #mW/um2
    
    Ispr = I*(scattering*10**12) # mW
    
    print('laser', laser, 'ratio', r, 'betha', betha, 'I', I, 'Temp', Temp)
    
    El = lambda_to_energy(laser) # excitation wavelength in eV
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
    
    size = 60
    desired = np.where(long_AS[-1]-size<long_AS)
    IAS = np.sum(AS[desired])
    
  #  print('AS', laser, long_AS[desired])
    
    desired2 = np.where(long_S<long_S[0]+size)
    IS = np.sum(S[desired2])
    
  #  print('S', laser, long_S[desired2])
    
    QYStokes = IS/(absorption[closer(long, laser)]*(10**12)*I)
    
    return IAS, IS, long, PL, I, QYStokes 

def open_data(paren_folder, n, laser, size_notch, To, T, K):

    folder = []

    for files in os.listdir(parent_folder):
        if files.endswith("%1s"%n):
            folder = os.path.join(parent_folder, files)

    print(folder)

    list_of_files = os.listdir(folder)
    list_of_files.sort()
    list_of_files = list_of_files[1:6]

    L = len(list_of_files)

    ratio = np.zeros(L)
    
    londa_spr = np.zeros(L)
    width_spr = np.zeros(L)
    
    IAS_list = np.zeros(L)
    IS_list = np.zeros(L)
    I_list = np.zeros(L)
    
    QYStokes_list   = np.zeros(L)
    Qfactor_list  = np.zeros(L)

   # plt.figure()
   # plt.title('laser:%s'%laser)
    
    for i in range(L):
        
        name = os.path.join(folder, list_of_files[i])
        data = np.genfromtxt(name, delimiter=',')

        wavelength = data[1:,0]
        abso = data[1:,3]
        sca = data[1:,2]
        extinction = data[1:,1]

        r = int(list_of_files[i].split('.')[0])/2

        ratio[i] = r
        
        wave = np.linspace(wavelength[0], wavelength[-1], 1000)
        scattering = np.interp(wave, wavelength, sca)
        absorption = np.interp(wave, wavelength, abso)
        exc = np.interp(wave, wavelength, extinction)
        
        lspr, width, Qfactor = properties(wave, scattering)
        
        print('r', r, 'lspr', lspr, 'width', width)
        
     #   plt.plot(wave, absorption, 'b')
     #   plt.plot(wave, scattering, 'r')
        
        IAS, IS, long_PL, PL, I, QYStokes = spectrum2(wave, scattering, absorption, laser, size_notch, To, T, K, r)
        
        IAS_list[i] = IAS
        
        IS_list[i] = IS
        
        I_list[i] = I
        
        QYStokes_list[i] = QYStokes
        
        Qfactor_list[i] = Qfactor
        
        if laser==532:
        
            plt.figure()
            plt.title('%s'%r)
          #  plt.plot(wave, absorption, 'b-')
          #  plt.plot(wave, scattering, 'r-')
            plt.plot(long_PL, PL, '--')
            plt.show()
        
    #    plt.plot(long_PL, PL, '--')
        
   # plt.show()
   # plt.xlim(500,750)
   # plt.ylim(0, 0.03)    
    
    return ratio, IAS_list, IS_list, I_list, wave, scattering, absorption, long_PL, PL, QYStokes_list, Qfactor_list

def properties(wavelength, signal):
    
    lspr = round(wavelength[np.argmax(signal)],2) #nm
    a = wavelength[np.where(signal >= np.max(signal)/2)]
    width = round(a[-1] - a[0], 2) #nm
    
    E = lambda_to_energy(lspr)
    
    Qfactor = E/width
    
    return lspr, width, Qfactor

#%%
    
    
plt.close('all')

parent_folder = r'C:\Users\lupau\OneDrive\Documentos\Luciana Martinez\Programa_Python\Simule Spectrum Growth'
parent_folder = os.path.normpath(parent_folder)

n = 1.33

laser = 550
size_notch = 20
To = 295
#I = 0.5  #mW/um2
K = 0.8 #025 #0.6 #W/Km
K = K*1000 #mW

power_bfp = 0.7  #mW
T0 = 0.47
waist = 0.340 #um

i = (2*T0)*power_bfp/np.pi*waist**2

print('irradiance 532 nm mW/um2', i)

l = np.arange(502, 700, 10)
rr_30 = []
rr_40 = []
rr_50 = []

I_30 =  [] 
I_40 =  []
I_50 =  []  

T = 373.5 #K

QYStokes_30 = []
Qfactor_30 = []

QYStokes_40 = []
Qfactor_40 = []

QYStokes_50 = []
Qfactor_50 = []

#plt.figure()
for laser in l:

    ratio, IAS_list, IS_list, I_list, wave, scattering, absorption, long_PL, PL, QYStokes_list, Qfactor_list = open_data(parent_folder, n, laser, size_notch, To, T, K)
    
    power = I_list*(np.pi*waist**2)/(2*T0)
    
    print('power bfp', power)
    
    rr_30.append(IAS_list[0])
    rr_40.append(IAS_list[2])
    rr_50.append(IAS_list[4])
    
    I_30.append(power[0])
    I_40.append(power[2])
    I_50.append(power[4])
    
    QYStokes_30.append(QYStokes_list[0])
    Qfactor_30.append(Qfactor_list[0])
    
    QYStokes_40.append(QYStokes_list[2])
    Qfactor_40.append(Qfactor_list[2])
    
    QYStokes_50.append(QYStokes_list[4])
    Qfactor_50.append(Qfactor_list[4])
    
   # plt.plot(ratio, IAS_list/IS_list,'o')
   # plt.plot(ratio, IAS_list, '.')
   
    
#plt.ylim(0, 0.08)
#plt.show()

plt.figure()
plt.title('Integral anti-Stoke')
plt.plot(l, rr_30, 'o', color ='C0')
plt.plot(l, rr_40, 'o', color = 'C2')
plt.plot(l, rr_50, 'o', color ='C4')
plt.ylim(0, 5)
plt.show()

plt.figure()
plt.title('power bfp')
plt.plot(l, I_30, 'o', color ='C0')
plt.plot(l, I_40, 'o', color = 'C2')
plt.plot(l, I_50, 'o', color ='C4')
plt.show()

plt.figure()
plt.title('Q')
plt.plot(Qfactor_30[3], QYStokes_30[3], 'o', color ='C0')
plt.plot(Qfactor_40[3], QYStokes_40[3], 'o', color = 'C2')
plt.plot(Qfactor_50[3], QYStokes_50[3], 'o', color ='C4')
plt.xlabel('Q-factor')
plt.ylabel('QY')
plt.show()


#%%

plt.close('all')

long = np.arange(400, 701, 1)
size_notch = 20
To = 293
I = 1
longSPR = 590
gammaSPR = 100
betha_532 = 50

laser = np.arange(450,660,10)

print(laser)

ratio = []
IAS_list = []
IS_list = []

for l in laser:
    
    Ispr, IAS, IS, r = spectrum(long, l, size_notch, To, betha_532, I, longSPR, gammaSPR)
    ratio.append(r)
    IAS_list.append(IAS)
    IS_list.append(IS)

plt.figure()
plt.plot(laser, ratio, 'o')
#plt.plot(laser, IS_list, '.')
plt.plot(laser, IAS_list, '*')
plt.plot(long, Ispr, 'k')
plt.ylim(0, 6)
plt.show()

#%%
laser = 592
longSPR_list = np.arange(540, 650, 10)

print(laser)

ratio = []
IAS_list = []
IS_list = []

for longSPR in longSPR_list:
    
    Ispr, IAS, IS, r = spectrum(long, laser, size_notch, To, betha_532, I, longSPR, gammaSPR)
    ratio.append(r)
    IAS_list.append(IAS)
    IS_list.append(IS)

plt.figure()
#plt.plot(longSPR_list, ratio, 'o')
#plt.plot(longSPR_list, IS_list, '.')
plt.plot(longSPR_list, IAS_list, '*')
#plt.ylim(0, 6)
plt.show()

#%%
laser = 532
longSPR = 550
gammaSPR_list = np.arange(50, 150, 10)

print(laser)

ratio = []
IAS_list = []
IS_list = []

for gammaSPR in gammaSPR_list:
    
    Ispr, IAS, IS, r = spectrum(long, laser, size_notch, To, betha_532, I, longSPR, gammaSPR)
    ratio.append(r)
    IAS_list.append(IAS)
    IS_list.append(IS)

plt.figure()
#plt.plot(longSPR_list, ratio, 'o')
#plt.plot(longSPR_list, IS_list, '.')
plt.plot(gammaSPR_list, IAS_list, '*')
#plt.ylim(0, 6)
plt.show()