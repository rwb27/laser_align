import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 2, 0.0001)
y = x**2 + np.random.random(x.size)

plt.plot(x, y)
plt.show()
print np.polyfit(x, y, 2, full=True)