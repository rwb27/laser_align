import numpy as np
import h5py
import matplotlib.pyplot as plt

dt = 1.34
fftsize = 3600
t = np.arange(fftsize)*dt


df = h5py.File(r'C:\Users\a-amb\OneDrive - University Of '
               r'Cambridge\2016-08-26-timedmmts.h5', 'r')
data = df["TimedMeasurements/TimedMeasurements_4/brightness_final"][:, ...]
print data.shape
print data[:, 0]

f = np.arange(fftsize)/(fftsize*dt)
Y = np.fft.fft(data[:, 4])
print f[:1448]
dat = np.fft.ifft(Y[:1448])
plt.plot(data[:, 0][:1448], dat)
plt.show()