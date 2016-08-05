import numpy as np

arrs = np.linspace(0, 10000, 100000)


@profile
def for_loop(arr):
    for item in arr:
        print item


@profile
def iterator(arr):
    for x in np.nditer(arr):
        print  x


@profile
def buffered(arr):
    for x in np.nditer(arr, flags=['buffered']):
        print x


for_loop(arrs)
