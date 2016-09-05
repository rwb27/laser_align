import matplotlib.pyplot as plt
import nplab

nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\2016-09-02-raster2d.h5", mode="r")
df = nplab.current_datafile()
data = df['RasterXY/RasterXY_0/brightness_final'][:, ...]
fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = data[:, 1], data[:, 2]
Z = data[:, 4]
ax.plot_trisurf(X, Y, Z)

plt.show()
