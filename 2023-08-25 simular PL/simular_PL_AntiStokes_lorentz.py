# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:11:52 2023

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

def spectrum(long, laser, size_notch, To, betha532, T, longSPR, gammaSPR):
    
    desired_AS = np.where(long < laser-int(size_notch/2) )
    desired_S = np.where(long > laser+ int(size_notch/2)  )
    
    long_AS = long[desired_AS]
    long_S =  long[desired_S]
    
    p =  longSPR, gammaSPR
    
    energyAS = lambda_to_energy(long_AS)
    
    Ispr = lorentz(long, *p)
    
    betha = int(betha532*(Ispr[closer(long, laser)]/Ispr[closer(long, longSPR)]))
    
    Temp = T
    
    I = round((T-To)/betha, 2)
       
    Ispr = I*Ispr
    
    print('laser', laser, 'betha', betha)
    
    El = lambda_to_energy(laser) # excitation wavelength in eV
    BE = bose(energyAS, El, Temp) 
    
    AS = Ispr[desired_AS]*BE
    S = Ispr[desired_S]
    
    PL = np.zeros(len(long))
    PL[desired_AS] = AS
    PL[desired_S] = S
    
    totalAS = np.sum(AS)*(long_AS[1]-long_AS[0])*El
    totalS = np.sum(S)*(long_S[1]-long_S[0])*El
  
    #step = 2
   # dAS = np.arange(long_AS[0], long_AS[-1]+step, step)
   # waveAS = (dAS[:-1] + dAS[1:])/2
    #energy_shift_AS = lambda_to_energy(waveAS) - lambda_to_energy(laser)
    
    step = 0.02
    eAS = lambda_to_energy(long_AS) - lambda_to_energy(laser)
    deAS = np.arange(eAS[-1], eAS[0]+step, step)
    energy_shift_AS = (deAS[:-1] + deAS[1:])/2
    waveAS = energy_to_lambda(energy_shift_AS + lambda_to_energy(laser))
    
    print('deAS', deAS)
    
    IAS = np.zeros(len(energy_shift_AS))
    
    for i in range(len(energy_shift_AS)-1):
        
     #   desired = np.where((long_AS > dAS[i])&((long_AS < dAS[i+1])))
        
        desired = np.where((eAS > deAS[i])&((eAS < deAS[i+1])))
        
        IAS[i] = np.sum(AS[desired])*step*El/betha
        
    #step = 2
   # dS = np.arange(long_S[0], long_S[-1]+step, step)
    #waveS = (dS[:-1] + dS[1:])/2 
   # energy_shift_S = lambda_to_energy(waveS) - lambda_to_energy(laser)
    
    step = 0.02
    eS = lambda_to_energy(long_S) - lambda_to_energy(laser)
    deS = np.arange(eS[-1], eS[0]+step, step)
    energy_shift_S = (deS[:-1] + deS[1:])/2
    waveS = energy_to_lambda(energy_shift_S + lambda_to_energy(laser))
    
    print('deS', deS)
    
    IS = np.zeros(len(energy_shift_S))
    
    for i in range(len(energy_shift_S)-1):
        
      #  desired = np.where((long_S > dS[i])&((long_S < dS[i+1])))
        desired = np.where((eS > deS[i])&((eS < deS[i+1])))
        
        IS[i] = np.sum(S[desired])*step*El/betha
    
    return I, betha, long, PL, totalAS, totalS, energy_shift_AS[:-1], waveAS[:-1], IAS[:-1], energy_shift_S[:-1], waveS[:-1], IS[:-1]

def power_law(Tarray, long, laser, size_notch, To, betha532, longSPR, gammaSPR):
    
    El = lambda_to_energy(laser)
    
    I, betha, long, PL, totalAS, totalS, energy_shift_AS, wave_As, IAS, energy_shift_S, wave_S, IS = spectrum(long, laser, size_notch, To, betha532, Tarray[0], longSPR, gammaSPR)
    
    I_array = np.zeros(len(Tarray))
    totalAS_array = np.zeros(len(Tarray))
    totalS_array = np.zeros(len(Tarray))
    
    matrix_AS = np.zeros((len(Tarray), len(IAS)))
    
    matrix_S = np.zeros((len(Tarray), len(IS)))
    
    matrix_PL = np.zeros((len(Tarray), len(PL)))
    
    I_array[0] = I
    matrix_AS[0, :] = IAS
    matrix_S[0, :] = IS
    matrix_PL[0,:] = PL
    
    plt.figure()
    plt.title('laser %s'%laser)
    plt.ylabel('Sum PL')
    plt.xlabel('Energy Shift (eV)')
    
    for i in range(len(Tarray)):
        
        T = Tarray[i]
        
        I, betha, long, PL, totalAS, totalS, energy_shift_AS, wave_AS, IAS, energy_shift_S, wave_S, IS = spectrum(long, laser, size_notch, To, betha532, T, longSPR, gammaSPR)
        
        I_array[i] = I
        totalS_array[i] = totalS
        totalAS_array[i] = totalAS
        matrix_AS[i,:] = IAS
        matrix_S[i,:] = IS
        matrix_PL[i,:] = PL
        
        plt.plot(energy_shift_AS, IAS, '-')
        plt.plot(energy_shift_S, IS, '-')
    
    plt.figure()
    plt.title('laser %s nm, betha %s Kum2/mW'% (laser, betha))
    plt.ylabel('PL')
    plt.xlabel('wavelength (nm)')
    
    for i in range(len(Tarray)):
        
        plt.plot(long, matrix_PL[i,:], label = 'T = %s K, I = %s mW/um2'%(Tarray[i], I_array[i]))
        
    plt.legend()
    
    x = np.log(I_array)
    pAS = np.zeros(matrix_AS.shape[1])
    QY_AS = np.zeros(matrix_AS.shape[1])
    pS = np.zeros(matrix_S.shape[1])
    QY_S = np.zeros(matrix_S.shape[1])
    
    for i in range(matrix_S.shape[1]):
        
        S = matrix_S[:, i]
        
        y = np.log(S)
        
        p = np.polyfit(x, y, 1)
        
        yfit = np.polyval(p, x)
        
        pS[i] = p[0]
        QY_S[i] = np.exp(p[1])
        
        
    for i in range(matrix_AS.shape[1]):
        
        AS = matrix_AS[:, i]
        
        y = np.log(AS)
        
        p = np.polyfit(x, y, 1)
        
        yfit = np.polyval(p, x)
        
        pAS[i] = p[0]
        QY_AS[i] = np.exp(p[1])
        

    return I_array, betha, totalS_array, totalAS_array, energy_shift_AS, wave_AS, pAS, QY_AS, energy_shift_S, wave_S, pS, QY_S
        

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out


def properties(wavelength, signal):
    
    lspr = round(wavelength[np.argmax(signal)],2) #nm
    a = wavelength[np.where(signal >= np.max(signal)/2)]
    width = round(a[-1] - a[0], 2) #nm
    
    E = lambda_to_energy(lspr)
    
    Qfactor = E/width
    
    return lspr, width, Qfactor


long = np.arange(400, 801, 1)
size_notch = 20
To = 295
longSPR = 550
gammaSPR = 70
betha532 = 60

Tarray = np.arange(300, 450, 25)

#ax3.set_ylim(0, 2)

lasers = [450, 550, 650]# np.arange(440, 640, 10) #[440, 550, 600, 650]
colors = ['b','green', 'r']

lspr_list = 550
gammaSPR_list = [50]
Qfactor = np.zeros(1)

plt.close('all')

fig3, ax3 = plt.subplots()
ax3.set_ylabel('QY Anti-Stokes')
ax3.set_xlabel('laser (nm)')

fig3b, ax3b = plt.subplots()
ax3b.set_ylabel('QY Stokes')
ax3b.set_xlabel('laser (nm)')


QS_550 = np.zeros(len(Qfactor))
QS_450 = np.zeros(len(Qfactor))

for j in range(len(gammaSPR_list)):

    gammaSPR = gammaSPR_list[j]
    
    fig1, ax1 = plt.subplots()
    ax1.set_ylabel('power law')
    ax1.set_xlabel('Energy Shift (eV)')

    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('QY')
    ax2.set_xlabel('Energy Shift (eV)')
    
    fig4, ax4 = plt.subplots()
    ax4.set_ylabel('QY')
    ax4.set_xlabel('longitud de onda (nm)')
    
    ax1.set_ylim(0, 6)
    ax2.set_ylim(-0.01, 0.1)#53)
    ax4.set_ylim(-0.01, 0.1)#53)
    
    QAS_total = np.zeros(len(lasers))
    QS_total = np.zeros(len(lasers))
    
     
    for i in range(len(lasers)):
        
        laser = lasers[i]
    
        I_array, betha, totalS_array, totalAS_array, energy_shift_AS, wave_AS, pAS, QY_AS, energy_shift_S, wave_S, pS, QY_S = power_law(Tarray, long, laser, size_notch, To, betha532, longSPR, gammaSPR)
        c = colors[i]
           
        ax1.plot(energy_shift_AS, pAS, 'o', color = c)
        ax1.plot(energy_shift_S, pS, 'o', color = c)
        
        ax2.plot(energy_shift_AS, QY_AS, 'o', color = c)
        ax2.plot(energy_shift_S, QY_S, 'o', color = c)
        
        QAS_total[i] = np.sum(QY_AS)
        QS_total[i] = np.sum(QY_S)
        
        ax4.plot(wave_AS, QY_AS, 'o', color = c)
        ax4.plot(wave_S, QY_S, 'o', color = c)
        
        print(laser, 'QY AS toltal', np.mean(totalAS_array/I_array)/betha, np.sum(QY_AS))
        print(laser, 'QY S toltal', np.mean(totalS_array/I_array)/betha, np.sum(QY_S))
        
        if laser == 550:
            print('550', i)
            QS_550[j] = QS_total[i] 
            
        if laser == 450:
            print('ok 450', i)
            QS_450[j] = QS_total[i]
            
    Qfactor[j] = round(lambda_to_energy(longSPR)/gammaSPR,4)
    
    ax4.plot(long,  lorentz(long, *[longSPR, gammaSPR]), '--', color = 'grey')
    
   # ax3.plot(lasers, QS_total, 'o')
    ax3.plot(lasers, QAS_total, '*', label = '%s'%gammaSPR)
    ax3b.plot(lasers, QS_total, 'o', label = '%s'%gammaSPR)
    
ax3.legend()
ax3b.legend()    

#%%

fig6, ax6 = plt.subplots()
ax6.plot(Qfactor, QS_550, 'go')
ax6.plot(Qfactor, QS_450, 'bo')

print(QS_450, QS_550)