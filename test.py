import numpy as np

a = np.array([[1000,2000,3000],[4000,5000,6000]])

it = np.nditer(a, flags=['multi_index'])
results = []
while not it.finished:
    results.append([it.multi_index[0], it.multi_index[1], it[0]])
    it.iternext()

print np.array(results)