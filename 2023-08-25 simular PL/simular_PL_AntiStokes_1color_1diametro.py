# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 19:28:00 2023

@author: lupau
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import re

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
   
    Temp = T
    
    I = round((T-To)/betha,4)
    
    Ispr = I*(scattering*10**12) # mW
    
    print('laser', laser, 'ratio', r, 'betha', betha, 'I', I, 'Temp', Temp)
    
    El = lambda_to_energy(laser) # excitation wavelength in eV
    BE = bose(energyAS, El, Temp) 
    
    AS = Ispr[desired_AS]*BE
    
    S = Ispr[desired_S]
    
  
    PL = np.zeros(len(long))
    PL[desired_S] = S
    PL[desired_AS] = AS
    
    size = 100
    desired = np.where(long_AS[-1]-size<long_AS)
    IAS = np.sum(AS[desired])
    
    desired2 = np.where(long_S<long_S[0]+size)
    IS = np.sum(S[desired2])
    
    return IAS, IS, long, PL, I, betha


def analyze_QY(parent_folder, medium, laser, size_notch, To, T_array, K, size_wave):
    
    name = "gold_nanoparticle_diameter_%s"%medium
    folder = os.path.join(parent_folder, name)

    list_of_files = os.listdir(folder)
    list_of_files.sort()
        
    name = os.path.join(folder, list_of_files[1])
    data = np.genfromtxt(name, delimiter=',')

    wavelength = data[1:,0]
    abso = data[1:,3]
    sca = data[1:,2]
    extinction = data[1:,1]

    r = int(list_of_files[1].split('.')[0])/2
    
    desired = np.where((wavelength > laser-int(size_wave/2)) & (wavelength < laser+int(size_wave/2)) )
    wavelength =  wavelength[desired]
    sca = sca[desired]
    abso = abso[desired]
   
    wave = np.linspace(wavelength[0], wavelength[-1], 1000)
    
    scattering = np.interp(wave, wavelength, sca)
    absorption = np.interp(wave, wavelength, abso)
    long = wave
    
    El = lambda_to_energy(laser) # excitation wavelength in eV
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    desired_S = np.where(long > laser+ int(size_notch/2) )
    
    long_AS = long[desired_AS]
    long_S =  long[desired_S]
   
    energyAS = lambda_to_energy(long_AS)
    
    betha = absorption[closer(long, laser)]/(4*np.pi*(r*10**-9)*K)# K.m2/mW
    
    betha = betha*10**12 # K.um2/mW
    
    betha = round(betha, 2)
    
    n = len(T_array)
    
    Sum_S = np.zeros(n)
    Sum_AS = np.zeros(n)
    I_array = np.zeros(n)
    
    for i in range(n):
        
        Temp = T_array[i]
        
        I = round((Temp-To)/betha,4)
        
        Ispr = I*(scattering*10**12) # mW
        
        #print('laser', laser, 'ratio', r, 'betha', betha, 'I', I, 'Temp', Temp)
        
        BE = bose(energyAS, El, Temp) 
        AS = Ispr[desired_AS]*BE
        S = Ispr[desired_S]
        
        I_array[i] = I
        Sum_AS[i] = np.sum(AS)
        Sum_S[i] = np.sum(S)
    
    plt.figure()
    plt.plot(I_array, Sum_S, 'ro')
    plt.plot(I_array, Sum_AS, 'bo')
    plt.show()
    
    plt.figure()
    
    x, yfit, pS = ajuste_lineal_log(I_array, Sum_S)
    print('Ajuste lineal S', pS[0], pS[1], np.exp(pS[1]) )
    plt.plot(x, yfit, 'ro-')
    
    x, yfit, pAS = ajuste_lineal_log(I_array, Sum_AS)
    print('Ajuste lineal AS', pAS[0], pAS[1], np.exp(pAS[1]) )
    plt.plot(x, yfit, 'bo-')
    
    QY_S = np.exp(pS[1]) #np.mean(Sum_S/I_array)
    QY_AS = np.exp(pAS[1]) # np.mean(Sum_AS/I_array)
    
    return betha, I_array, QY_AS, QY_S, pAS, pS

def analyze_spec(long, signal, step_size):
    
    n = int((long[-1] - long[0])/step_size)
    
    d = np.linspace(long[0], long[-1], n)

    wave = (d[0:-1]+ d[1:])/2
    Spec_Sum = np.zeros(len(wave))
    
    for i in range(len(wave)):
        
         desired = np.where((d[i]<=long)&(d[i+1]>=long))
         
         Spec_Sum[i] = np.sum(signal[desired])  
    
    return wave, Spec_Sum

def analyze_QY_specs(parent_folder, medium, laser, size_notch, To, T_array, K, step_size, size_wave):
    
    name = "gold_nanoparticle_diameter_%s"%medium
    folder = os.path.join(parent_folder, name)

    list_of_files = os.listdir(folder)
    list_of_files.sort()
        
    name = os.path.join(folder, list_of_files[1])
    data = np.genfromtxt(name, delimiter=',')

    wavelength = data[1:,0]
    abso = data[1:,3]
    sca = data[1:,2]
    extinction = data[1:,1]

    r = int(list_of_files[1].split('.')[0])/2
    
    desired = np.where((wavelength > laser-int(size_wave/2)) & (wavelength < laser+int(size_wave/2)) )
    wavelength =  wavelength[desired]
    sca = sca[desired]
    abso = abso[desired]
   
   
    wave = np.linspace(wavelength[0], wavelength[-1], 1000)
    scattering = np.interp(wave, wavelength, sca)
    absorption = np.interp(wave, wavelength, abso)
    long = wave
    
    El = lambda_to_energy(laser) # excitation wavelength in eV
    
    desired_AS = np.where(long < laser-int(size_notch/2))
    desired_S = np.where(long > laser+ int(size_notch/2))
    
    long_AS = long[desired_AS]
    long_S =  long[desired_S]
   
    energyAS = lambda_to_energy(long_AS)
    
    Abs = absorption[closer(long, laser)]
    
    betha = Abs/(4*np.pi*(r*10**-9)*K)# K.m2/mW
    
    betha = betha*10**12 # K.um2/mW
    
    betha = round(betha, 2)
    
    n = len(T_array)
    I_array = np.zeros(n)
    
    lenwAS = int( (long_AS[-1]-long_AS[0])/step_size)- 1
    lenwS = int( (long_S[-1]-long_S[0])/step_size ) - 1
    
    matrixAS = np.zeros((lenwAS+1, n))
    matrixS = np.zeros((lenwS+1, n))
    
    for i in range(n):
        
        Temp = T_array[i]
        
      #  Temp = To + I*betha
        
        I = round((Temp-To)/betha, 4)
        
        I_array[i] = I
        
        Ispr = I*(scattering*10**12) # mW
        
        #print('laser', laser, 'ratio', r, 'betha', betha, 'I', I, 'Temp', Temp)
        
        BE = bose(energyAS, El, Temp) 
        AS = Ispr[desired_AS]*BE
        S = Ispr[desired_S]
        
        waveAS, Sum_AS_wave = analyze_spec(long_AS, AS, step_size)
        waveS, Sum_S_wave = analyze_spec(long_S, S, step_size)
        
        matrixAS[0, i] = I
        matrixS[0, i] = I
        
        for j in range(1,lenwAS):
            
            matrixAS[j, i] = Sum_AS_wave[j]
            
        for k in range(1,lenwS):
            
            matrixS[k, i] = Sum_S_wave[k]
            
    pAS_wave = np.zeros(lenwAS)
    pS_wave = np.zeros(lenwS)
    
    QAS_wave = np.zeros(lenwAS)
    QS_wave = np.zeros(lenwS)
    
    for l in range(1, lenwAS):
        
        Sum_AS = matrixAS[l, :]
        
        x, yfit, pAS = ajuste_lineal_log(I_array, Sum_AS)
       # print('Ajuste lineal AS', pAS[0], pAS[1], np.exp(pAS[1]) )
        
        pAS_wave[l] = pAS[0]
        QAS_wave[l] = np.exp(pAS[1])
        
 #   print('Ajuste AS', pAS_wave)
    
    for l in range(1, lenwS):
        
        Sum_S = matrixS[l, :]
        
        x, yfit, pS = ajuste_lineal_log(I_array, Sum_S)
       # print('Ajuste lineal AS', pAS[0], pAS[1], np.exp(pAS[1]) )
        
        pS_wave[l] = pS[0]
        QS_wave[l] = np.exp(pS[1])
        
#    print('Ajuste S', pS_wave)
    
    wexc = laser
    energy_shiftAS = lambda_to_energy(waveAS)-lambda_to_energy(wexc)
    energy_shiftS = lambda_to_energy(waveS)-lambda_to_energy(wexc)
    
    return betha, energy_shiftAS[1:], pAS_wave[1:], QAS_wave[1:], energy_shiftS[1:], pS_wave[1:], QS_wave[1:]

def ajuste_lineal_log(x0, y0):
    
    x = np.log(x0)
    y = np.log(y0)
    
    p = np.polyfit(x, y, 1)
    yfit = np.polyval(p, x)
    
    return x, yfit, p

def open_data(parent_folder, medium, laser, size_notch, To, T, K, plt_bool):
    
    name = "gold_nanoparticle_diameter_%s"%medium
    
    folder = os.path.join(parent_folder, name)

    print(folder)

    list_of_files = os.listdir(folder)
    list_of_files.sort()

    L = len(list_of_files)
    
    i = 1
        
    name = os.path.join(folder, list_of_files[i])
    data = np.genfromtxt(name, delimiter=',')

    wavelength = data[1:,0]
    abso = data[1:,3]
    sca = data[1:,2]
    extinction = data[1:,1]

    r = int(list_of_files[i].split('.')[0])/2
   
    wave = np.linspace(wavelength[0], wavelength[-1], 1000)
    scattering = np.interp(wave, wavelength, sca)
    absorption = np.interp(wave, wavelength, abso)
    exc = np.interp(wave, wavelength, extinction)

    IAS, IS, long_PL, PL, I, betha = spectrum2(wave, scattering, absorption, laser, size_notch, To, T, K, r)
   
    if plt_bool:
        
        fig, ax = plt.subplots()
        plt.title('laser:%s,ratio:%s,medium:%s'%(laser, r, medium))
        ax.plot(wave, absorption*10**18, 'b')
        ax.plot(wave, scattering*10**18, 'r')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Abs, Scattering (nm2)')
        ax.set_ylim(0, 30000)
    
    return r, IAS, IS, long_PL, PL, I, betha

#%%


plt.close('all')

base_folder = r'G:\Mi unidad'
parent_folder = os.path.join(base_folder, r'2023-08-25 simular PL')
parent_folder = os.path.normpath(parent_folder)

m = ["glass", "water", "air"]

plt_bool = False

#n = 1.33
ns = np.array([1.54, 1.33, 1.00])  #glass, water, air

#laser = 592
size_notch = 20
To = 295
#I = 0.5  #mW/um2
#K = 0.6 #W/Km
#K = K*1000 #mW/Km

T = 350 #K

Ks = np.array([0.8, 0.6, 0.025]) #glass, water, air
Ks = Ks*1000

lasers = np.arange(430, 830, 30)

IAS_list = np.zeros(( 3,len(lasers)))
bethas = np.zeros(( 3,len(lasers)))
I_list = np.zeros(( 3,len(lasers)))
    
fig3, ax3 = plt.subplots()
ax3.set_ylabel('F')
ax3.set_xlabel('laser exc (nm)')
#ax3.set_ylim(0, 1.2)

fig4, ax4 = plt.subplots()
ax4.set_ylabel('betha')
ax4.set_xlabel('laser exc (nm)')

fig5, ax5 = plt.subplots()
ax5.set_ylabel('Irradiance')
ax5.set_xlabel('laser exc (nm)')

for l in range(len(lasers)):
    
    laser = lasers[l]
    
   #  fig2, ax2 = plt.subplots()

    for i in range(3):
        
        medium = m[i]
        K = Ks[i]
    
        ratio, IAS, IS, long_PL, PL, I, betha = open_data(parent_folder, medium, laser, size_notch, To, T, K, plt_bool)
    
        bethas[i, l] = betha
        IAS_list[i,l] = IAS
        I_list[i,l] = I
        
      #  ax2.plot(long_PL, PL, label = 'n:%s, irradiance:%s'%(medium,I))
        
   #  ax2.set_xlim(400,800)
   #  ax2.set_ylim(0, 0.06)   
   #  ax2.legend()
    
for i in range(3):
    ax3.plot(lasers, IAS_list[i,:]/(T-To), 'o', label = "medium =%s"%m[i])
    ax4.plot(lasers, bethas[i,:], 'o', label = "medium =%s"%m[i])
    ax5.plot(lasers, I_list[i,:], 'o', label = "medium =%s"%m[i])
ax3.legend()
ax4.legend()
ax5.legend()


#%%


medium = "water"

laser = 532
size_notch = 20
To = 295

T_array = np.arange(300, 450, 10) #K

K = 0.6 #W/Km
K = K*1000

laser = 532
size_wave = 120

betha, I_array, QY_AS, QY_S, pAS, pS = analyze_QY(parent_folder, medium, laser, size_notch, To, T_array, K, size_wave)

print('QY', QY_AS, QY_S)
print('F', QY_AS/betha, QY_S/betha)
#%%

medium = "water"
size_notch = 16
To = 295
K = 0.6 #W/Km
K = K*1000

T_array = np.arange(300, 400, 10)

step_size = 5
size_wave = 150

fig, ax= plt.subplots()
ax.set_ylabel('Power law')
ax.set_xlabel('Energy shift (eV)')

fig2, ax2= plt.subplots()
ax2.set_ylabel('F')
ax2.set_xlabel('Energy shift (eV)')

betha,energy_shiftAS, pAS_wave, QAS_wave, energy_shiftS, pS_wave, QS_wave = analyze_QY_specs(parent_folder, medium, 532, size_notch, To, T_array, K, step_size, size_wave)

print('sum all QY', np.sum(QAS_wave), np.sum(QS_wave))
print('F', np.sum(QAS_wave)/betha, np.sum(QS_wave)/betha)

ax.plot(energy_shiftAS, pAS_wave, 'o',color = 'green')
ax.plot(energy_shiftS, pS_wave, 'o',color ='green')
ax2.plot(energy_shiftAS, QAS_wave/betha, 'o',color = 'green')
ax2.plot(energy_shiftS, QS_wave/betha, 'o',color ='green')


betha,energy_shiftAS, pAS_wave, QAS_wave, energy_shiftS, pS_wave, QS_wave  = analyze_QY_specs(parent_folder, medium, 592, size_notch, To, T_array, K, step_size, size_wave)


ax.plot(energy_shiftAS, pAS_wave,'o', color ='orange')
ax.plot(energy_shiftS, pS_wave,'o', color ='orange')
ax2.plot(energy_shiftAS, QAS_wave/betha, 'o',color = 'orange')
ax2.plot(energy_shiftS, QS_wave/betha, 'o',color ='orange')

print('sum all QY', np.sum(QAS_wave), np.sum(QS_wave))
print('F', np.sum(QAS_wave)/betha, np.sum(QS_wave)/betha)

betha,energy_shiftAS, pAS_wave, QAS_wave, energy_shiftS, pS_wave, QS_wave  = analyze_QY_specs(parent_folder, medium, 642, size_notch, To, T_array, K, step_size, size_wave)


ax.plot(energy_shiftAS, pAS_wave,'o', color ='r')
ax.plot(energy_shiftS, pS_wave,'o', color ='r')
ax2.plot(energy_shiftAS, QAS_wave/betha, 'o',color = 'r')
ax2.plot(energy_shiftS, QS_wave/betha, 'o',color ='r')

print('sum all QY', np.sum(QAS_wave), np.sum(QS_wave))
print('F', np.sum(QAS_wave)/betha, np.sum(QS_wave)/betha)

medium = "air"
K = 0.025 #W/Km
K = K*1000

betha,energy_shiftAS, pAS_wave, QAS_wave, energy_shiftS, pS_wave, QS_wave  = analyze_QY_specs(parent_folder, medium, 532, size_notch, To, T_array, K, step_size, size_wave)



ax.plot(energy_shiftAS, pAS_wave, '*',color = 'C1')
ax.plot(energy_shiftS, pS_wave, '*',color ='C1')
ax2.plot(energy_shiftAS, QAS_wave/betha, '*',color = 'C1')
ax2.plot(energy_shiftS, QS_wave/betha, '*',color ='C1')

print('sum all QY', np.sum(QAS_wave), np.sum(QS_wave))
print('F', np.sum(QAS_wave)/betha, np.sum(QS_wave)/betha)
