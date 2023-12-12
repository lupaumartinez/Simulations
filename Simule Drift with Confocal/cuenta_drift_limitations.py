#%% Cuenta drift

size_scan = 800 #nm
number_pixel_array = np.arange(8,21)
pixel_size_array = size_scan/number_pixel_array
velocity_drift_max = 0.5 #nm/s
exposure_time_array = np.array([0.5, 0.8, 1, 2]) #s

time_total = number_pixel_array**2#*exposure_time_array
drift_max_per_pixel = velocity_drift_max*time_total/pixel_size_array

plt.figure()
plt.axhspan(0.5, 1, facecolor='0.5', alpha=0.5)
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[0], '--', label = 't_{exp}_{px} = 0.5 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[1], '--', label = 't_{exp}_{px} = 0.8 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[2], '--', label = 't_{exp}_{px} = 1.0 s')
plt.plot(number_pixel_array, drift_max_per_pixel*exposure_time_array[3], '--', label = 't_{exp}_{px} = 2.0 s')
plt.legend()
plt.xlabel('Number of pixels')
plt.ylabel('Drift max / pixel size')
plt.title('Size scan = 800 nm, Vel. Drift = 0.5 nm/s')
plt.ylim(0, 3)
plt.xlim(7.5, 20.5)

#plt.savefig(parent_folder, 'confocal_limitations_by_drift.png')
plt.close()