
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#1 pixel = 60 nm
#1 frame de video = 30 s


def image_NP(tamaño, x_NP, y_NP):

	A = 10
	b = 2 #pixel

	i = 0
	j = 0

	I_NP = np.zeros((tamaño, tamaño))

	for i  in range(tamaño):
		for j  in range(tamaño):
			I_NP[i, j] = A*np.exp(   -( (i-x_NP)**2 + (j-y_NP)**2 )/(2*b)**2  )

	return I_NP


def image_NP_minimum(tamaño, x_NP, y_NP):

	I_NP = image_NP(tamaño, x_NP, y_NP)

	I_NP_minimum  = stats.norm.pdf(I_NP)

	return I_NP_minimum


def video_NP(tamaño, tiempo):

	wx = np.pi/5
	wy = np.pi/5

	t = 0
	video_I_NP = []
	x = []
	y = []

	xo = tamaño/2
	yo = tamaño/2

	while t < tiempo:
		x_NP = xo + np.sin(wx*t) - 5*t
		y_NP = yo + np.sin(wy*t) - 2*t
		I_NP = image_NP(tamaño, x_NP, y_NP)
		video_I_NP.append(I_NP)
		x.append(x_NP)
		y.append(y_NP)
		t = t + 1

	return video_I_NP, x, y


#video = video_NP(50, 4)
#plt.imshow(video[0])
#plt.show()

