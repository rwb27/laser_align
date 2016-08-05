import h5py
import numpy as np
import matplotlib.pyplot as plt

df = h5py.File(r"C:\Users\a-amb\OneDrive - University Of "
               r"Cambridge\2016-08-04.h5", mode="r")
data = df['Tiled/brightness_results_1']
thing = np.delete(data, 2,1)
thing[:, 0:2] += 300
thing[:, :2] /= 100
reshaped = np.zeros((7, 7))
for row in thing:
    reshaped[row[0], row[1]] = row[2]
plt.imshow(reshaped)
plt.show()
