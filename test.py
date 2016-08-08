import numpy as np

x = np.linspace(-1000 / 2., 1000/ 2., 11)
y = x


def positions_maker(x, y, initial_pos=np.array([0, 0, 0])):
    i = 0
    j = 0
    while i < x .size:
        while j < y.size:
            yield np.array([x[i], y[j], 0]) + initial_pos
            j += 1
        i += 1

pos = positions_maker(x, y)
try:
    while True:
        print next(pos)
except StopIteration:
    print "hello"