# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:35:28 2023

@author: lupau
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def irradiance(power_bfp, t, omega):
    
    I = 2*power_bfp*t/(np.pi*omega**2)
    
    return round(I,2)

def betha(sigma_abs, radius_eq, gamma):
    
    #sigma_abs m2
    
    r = radius_eq*10**-9 # m
    
    b = sigma_abs*gamma/(4*np.pi*r)  #   K m2/mW
    
    b = b*10**12  #   K um2/mW
    
    return round(b,2)

def temperature_general(distance, sigma_abs, I, radius_eq,  gamma_m, R_k):
    
    r = radius_eq*10**-9 #m
    g_k = R_k/r #Km/mW 
    L_k = (R_k/gamma_m)*10**9 #nm
        
    print('gamma medium', gamma_m, 'gk #Km/mW ', g_k, 'length k nm', L_k)
    
    b =  betha(sigma_abs, radius_eq, gamma_m)
    print('betha', b)
    
    To = round(I*b,2) #K
    
    print('T sin kapitza', To)
    
    print('total T on NP', To*(L_k/radius_eq) + To)
    
    temperature = np.zeros(distance.shape[0])
    
    for i in range(0, distance.shape[0]):
        
        if np.abs(distance[i]) > (radius_eq):
            temperature[i] = To*radius_eq/np.abs(distance[i])
    
        if np.abs(distance[i]) <= radius_eq:
            
            temperature[i] = To*(L_k/radius_eq) + To
    
    return temperature, To, L_k

def temperature_general2(distance, sigma_abs, I, radius_eq,  gamma_m, R_k):
    
    r = radius_eq*10**-9 #m
    g_k = R_k/r #Km/mW 
    L_k = (R_k/gamma_m)*10**9 #nm
    
    ks = round((1/gamma_m)/1000,2)
    k = 308.2  # Au W/m/K
    factor = round(ks/(2*k),6)
    print('conductividad termica', ks, 'del Au', k, 'cociente', factor)
        
    print('gamma medium', gamma_m, 'gk #Km/mW ', g_k, 'length k nm', L_k)
    
    b =  betha(sigma_abs, radius_eq, gamma_m)
    print('betha', b)
    
    To = round(I*b,2) #K
    
    print('T sin kapitza', To)
    
    print('total T on NP', To*(L_k/radius_eq) + To)
    
    temperature = np.zeros(distance.shape[0])
    
    for i in range(0, distance.shape[0]):
        
        if np.abs(distance[i]) > (radius_eq):
            temperature[i] = To*radius_eq/np.abs(distance[i])
    
        if np.abs(distance[i]) <= radius_eq:
            
            temperature[i] = To*( 1 + factor*(1- (distance[i]/radius_eq)**2) + (L_k/radius_eq) )
    
    return temperature, To, L_k

if __name__ == '__main__':
    
    save_folder = r'C:\Ubuntu_archivos\Printing\Tesis Capitulos\Tesis Capitulo Nanothermometry'
    
    plt.close('all')
    
    Tamb = 25
    
    # Au in W/m/K

    sigma_abs = 1.804*1*10**-14 #m2 #  #1.80*1e-14 #      m2 para una np de oro de 80 nm 532 agua,  #1.804 de oro 80nm
    k_s = 0.58 # 0.8 water/glass from Setoura et al 2013, in W/m/K
    gamma_m = 1/k_s #Km/W
    gamma_m = gamma_m/1000 #Km/mW
    
    power_bfp  = 0.7 #mW
    t = 0.47
    omega = 0.240 # um tamaño de la PSF 
    radius_eq = 40 # nm radio de la NP
    
    R_k =  8.3*10**-12     #Km2/mW   water
    #were 12 K ⋅ m2/GW for the Au/Glass41 interface, 8.3 (red) or 4.0 (blue) K ⋅ m2/GW for the Au/water42 interface
    
    I = irradiance(power_bfp, t, omega) # mW/um^2
    
    print('irradiance', I)
    
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams ["axes.labelsize"] = 24
    plt.rcParams['xtick.labelsize']=24
    plt.rcParams['ytick.labelsize']=24
    
    plt.figure(figsize = (10,6))
    plt.xlabel('Radial coordinate')
    plt.ylabel(u'Temperature')
    
    plt.vlines(150/radius_eq, 0.266, 1.4, linestyle = '--', color = 'C0', linewidth = 3,alpha = 0.6)
    plt.vlines(-150/radius_eq, 0.266, 1.4, linestyle = '--',color = 'C0', linewidth = 3,alpha = 0.6)
    
    plt.vlines(-(10+radius_eq)/radius_eq, 0.8, 1.4, linestyle = '--',color = 'green', linewidth = 3, alpha = 0.6)
    plt.vlines((10+radius_eq)/radius_eq, 0.8, 1.4, linestyle = '--',color = 'green',linewidth = 3, alpha = 0.6)
    
    s = 0.1
    distance = np.arange(-20*radius_eq, 20*radius_eq +s, s) #nm
    
    DT, To,L_k = temperature_general2(distance, sigma_abs, I, radius_eq, gamma_m, R_k)
    r = round(L_k/radius_eq,2)
    plt.plot(distance/radius_eq, DT/To, 'r-', linewidth = 3, alpha = 0.6, label='$l_{th}/a$ ≈ % s'% r)
    
    j = np.min(DT[np.where((distance < (10+radius_eq)) & (distance > -(10+radius_eq)))])/To
    
    print(j)
    
  #  ome = omega*1000
  #  desired_xvol = np.where((distance >= - ome/2) & (distance <= ome/2))
  #  x_vol = distance[desired_xvol]
  #  y_vol = DT[desired_xvol]
  #  y_vol_mean = np.mean(y_vol)
  #  plt.hlines(y_vol_mean/To, xmin = round(x_vol[0]/radius_eq,2), xmax= round(x_vol[-1]/radius_eq), color = 'orange')
  #  plt.plot(x, max(DT)*np.ones(len(x)) + Tamb, '-r')
    
    DT, To,L_k = temperature_general2(distance, sigma_abs, I, radius_eq, gamma_m, 0)
    
    r = round(L_k/radius_eq,2)
    x = np.arange(-radius_eq, radius_eq+1, 1)
    plt.plot(distance/radius_eq, DT/To, 'k-', linewidth = 3, alpha = 0.6, label='$l_{th}/a$ ≈ % s'% r)
    
    plt.legend(loc = 'upper left', fontsize = 18)
    plt.xlim(-10, 10)
    plt.ylim(0, 1.4)
    plt.xticks(np.arange(-10, 12, 2))
    plt.yticks(np.arange(0.0, 1.6, 0.2))
    name = os.path.join(save_folder, 'figure temperature NP lines.png')
    plt.savefig(name, dpi = 400, bbox_inches='tight')
    
    plt.close()
      
  #  ome = omega*1000
  #  desired_xvol = np.where((distance <= ome/2) & (distance > radius_eq))
  ##  x_vol = distance[desired_xvol]
   # y_vol = DT[desired_xvol]
   # y_vol_mean = np.mean(y_vol)
   # x_vol_mean = ( x_vol[-1]+ x_vol[-1] - ome)/2
 #   plt.hlines(y_vol_mean/To, xmin = round(x_vol[0]/radius_eq,2), xmax= round(x_vol[-1]/radius_eq,2), color = 'grey')
 #   plt.plot(x_vol_mean/radius_eq, y_vol_mean/To, 'o', color ='grey')
    
  #  step = 10
  #  while ome + step < radius_eq + ome:
     #   step = step + 10
     #   desired_xvol = np.where((distance <= ome/2 + step) & (distance > radius_eq))
     #   x_vol = distance[desired_xvol]
     #   y_vol = DT[desired_xvol]
     #   y_vol_mean = np.mean(y_vol)
     #   x_vol_mean = ( x_vol[-1]+ x_vol[-1] - ome)/2
     #   plt.hlines(y_vol_mean/To, xmin = round(x_vol[0]/radius_eq,2), xmax= round(x_vol[-1]/radius_eq,2), color = 'grey')
     #   plt.plot(x_vol_mean/radius_eq, y_vol_mean/To, 'o', color ='grey')
        
#     for i in range(10):
        
 #        s = i*50
   #      desired_xvol = np.where((distance <= radius_eq + s + ome) & (distance > radius_eq + s))
     #    x_vol = distance[desired_xvol]
      #   y_vol = DT[desired_xvol]
       #  y_vol_mean = np.mean(y_vol)
        # x_vol_mean = (x_vol[0] + x_vol[-1])/2
        # plt.hlines(y_vol_mean/To, xmin = round(x_vol[0]/radius_eq,2), xmax= round(x_vol[-1]/radius_eq,2), color = 'grey')
        # plt.plot(x_vol_mean/radius_eq, y_vol_mean/To, 'o', color ='grey')
        
    
    
    #%%
    
    plt.figure(figsize = (2,2))
  #  plt.xlabel('Radial coordinate')
  #  plt.ylabel(u'Temperature')
    
    s = 0.01
    distance = np.arange(-radius_eq, radius_eq +s, s) #nm
    
    print(distance)
    
    DT, To,L_k = temperature_general2(distance, sigma_abs, I, radius_eq, gamma_m, 0)
    r = round(L_k/radius_eq,1)
    plt.plot(distance/radius_eq, DT/To, 'k-', linewidth = 3, alpha = 0.6)#, label='$l_{th}/a$ ≈ % s'% r)
    
    #plt.legend(fontsize = 16)
    plt.xlim(-1, 1)
    plt.ylim(1 - 0.001, 1 +0.002)
    
    name = os.path.join(save_folder, 'figure temperature NP zoom.png')
    plt.savefig(name, dpi = 400, bbox_inches='tight')
    plt.close()
