import matplotlib.pyplot as plt
import nplab
import numpy as np

nplab.datafile.set_current(r"C:\Users\a-amb\OneDrive - University Of "
                           r"Cambridge\3d_raster.hdf5")
df = nplab.current_datafile()

data = df['RasterXYZ/RasterXYZ_0/brightness_final'][:, ...]

unique = np.unique(data[:, 3])
for value in unique:
    subdata = data[np.where(data[:, 3] == value)]
    plt.figure()
    plt.scatter(subdata[:, 1], subdata[:, 2], subdata[:, 4])
    print np.max(subdata[:, 4])
    plt.show()
