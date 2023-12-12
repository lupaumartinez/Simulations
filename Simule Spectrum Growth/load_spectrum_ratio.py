
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

def closer(x,value):
    # returns the index of the closest element to value of the x array
    out = np.argmin(np.abs(x-value))
    return out

def lorentz(x, *p):
    # Lorentz fitting function with an offset
    # gamma = FWHM
    # I = amplitude
    # x0 = center
    pi = np.pi
    I, gamma, x0, C = p
    return (1/pi) * I * (gamma/2)**2 / ((x - x0)**2 + (gamma/2)**2) + C

def fit_lorentz(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)

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

    lorentz_fitted = lorentz(wavelength_S, *best_lorentz)
    r2_coef_pearson = calc_r2(spectrum_S, lorentz_fitted)

    full_lorentz_fitted = lorentz(wavelength, *best_lorentz)
    londa_max_pl = best_lorentz[2]
    width_pl = best_lorentz[1]

    return full_lorentz_fitted, londa_max_pl, width_pl

def open_data(paren_folder, n):

    folder = []

    for files in os.listdir(parent_folder):
        if files.endswith("%1s"%n):
            folder = os.path.join(parent_folder, files)

    print(folder)

    list_of_files = os.listdir(folder)
    list_of_files.sort()

    L = len(list_of_files)

    ratio = np.zeros(L)

    sigma_abs_532 = np.zeros(L)
    londa_spr = np.zeros(L)
    width_spr = np.zeros(L)
    max_spr  = np.zeros(L)
    max_spr_exc  = np.zeros(L)
    plt.figure()
    
    for i in range(L):
        
        name = os.path.join(folder, list_of_files[i])
        data = np.genfromtxt(name, delimiter=',')

        wavelength = data[1:,0]
        abso = data[1:,3]
        sca = data[1:,2]
        extinction = data[1:,1]

        ratio[i] = int(list_of_files[i].split('.')[0])/2
        
        wave = np.linspace(wavelength[0], wavelength[-1], 100)
        scattering = np.interp(wave, wavelength, sca)
        absorption = np.interp(wave, wavelength, abso)
        exc = np.interp(wave, wavelength, extinction)  
        
        out = closer(wave, 532)
        sigma_abs_532[i] = absorption[out]

        #full_lorentz_fitted, londa_max, width = fit_spr(scattering/max(scattering), wave, 530, 600)

        #londa_spr[i] = londa_max
        #width_spr[i] = width
        
        max_spr[i] = wave[np.argmax(scattering)]
        max_spr_exc[i] = wave[np.argmax(exc)]

        plt.plot(wave, scattering/max(scattering), label = '%2d'%ratio[i])
       # plt.plot(wave, full_lorentz_fitted, '--k')

    #plt.axvline(532, color ='g')
    plt.legend()
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Absorption [nm^{2}]')
    plt.show()


    return ratio, sigma_abs_532, max_spr, max_spr_exc


parent_folder = r'C:\Users\lupau\OneDrive\Documentos\Luciana Martinez\Programa_Python\Simule Spectrum Growth'
parent_folder = os.path.normpath(parent_folder)

n = 1.33

ratio, sigma_abs_532, max_spr, max_spr_exc = open_data(parent_folder, n)

plt.figure()
plt.plot(ratio, sigma_abs_532, 'o')
plt.xlabel('Ratio [nm]')
plt.ylabel('Sigma Abs 532 [nm^{2}]')
plt.show()

plt.figure()
plt.plot(ratio, max_spr, 'o')
plt.plot(ratio, max_spr_exc, '*')
plt.xlabel('Ratio [nm]')
plt.ylabel('Wavelength SPR [nm]')
plt.show()


#%% AJUSTE POLINOMICO

npol = 2
x = np.linspace(max_spr[0], max_spr[-1], 40)
p = np.polyfit(max_spr, ratio, npol)
poly = np.polyval(p, x)

npol = 2
p2= np.polyfit(max_spr_exc, ratio, npol)
poly2 = np.polyval(p2, x)

plt.figure()
#plt.plot(londa_spr, ratio, 'o')
plt.plot(max_spr, ratio, 'o')
plt.plot(x, poly, '*-')
plt.plot(max_spr_exc, ratio, 'o')
plt.plot(x, poly2, '*-')
plt.ylabel('Ratio [nm]')
plt.xlabel('Wavelength SPR Mie [nm]')
plt.show()

name = os.path.join(parent_folder,  'Simulation_max_wavelength_index_%s.txt'%n)
data = np.array([x, poly, poly2]).T
header_txt = 'Max wavelength (nm), Ratio_Sca, Ratio_Exc'
np.savetxt(name, data, header = header_txt)

#%%

npol = 3
x = np.linspace(ratio[0], ratio[-1], 40)
p = np.polyfit(ratio, sigma_abs_532, npol)
poly = np.polyval(p, x)

plt.figure()
#plt.plot(londa_spr, ratio, 'o')
plt.plot(ratio, sigma_abs_532,  'o')
plt.plot(x, poly, '*-')
plt.xlabel('Ratio [nm]')
plt.ylabel('Abs 532 Mie [nm]')
plt.show()

name = os.path.join(parent_folder, 'Simulation_abs532_index_%s.txt'%n)
data = np.array([x, poly]).T
header_txt = 'Ratio, Abs532nm'
np.savetxt(name, data, header = header_txt)




