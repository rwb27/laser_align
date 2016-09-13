import numpy as np
import matplotlib.pyplot as plt
import time

plt.ion()
fig, axes = plt.subplots(1, 3, sharey=True)
axes[0].plot([1,2,3],[4,5,6])
plt.show()
plt.show()
axes[1].plot([1,2,3],[4,5,6])
plt.ioff()

