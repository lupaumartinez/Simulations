import numpy as np
import matplotlib.pyplot as plt

# Power Sensor BS

meas_pow_bfp = 1 #mW
pow_BS = round((meas_pow_bfp/0.6)*0.4, 3)

exposure_time = 0.8
pixel_time = exposure_time + 0.1 #s
number_pixel = 12
confocal_time = pixel_time*number_pixel**2

confocal_ph = 10
autofoco = 3
total_time_NP = autofoco + confocal_ph + confocal_time

number_NP = 10
total_time = 10*total_time_NP

rate_I = 1/12 #Hz
rate_air = 1/(10*60)

t = np.arange(0, total_time, 0.300)
I = pow_BS + 0.01*pow_BS*np.sin(rate_I*t*np.pi) + 0.025*pow_BS*np.sin(rate_air*t*np.pi)

t_NP = np.arange(autofoco+confocal_ph, total_time_NP, 0.300)
I_NP = pow_BS + 0.01*pow_BS*np.sin(rate_I*t_NP*np.pi) + 0.025*pow_BS*np.sin(rate_air*t_NP*np.pi)

mean_power_experiment= np.mean(I)
mean_power_confocal = np.mean(I_NP)

plt.figure()
plt.plot(t, I, '-k')
plt.plot(t_NP, I_NP, '--g')
plt.xlabel('Experiment Time (s)')
plt.ylabel('Power BS (mW)')
plt.axhline(mean_power_experiment, color = 'k')
plt.axhline(mean_power_confocal, color = 'g')
plt.show()