import h5py


def printer(*args):
    print args


datfile = h5py.File('dat.hdf5', 'r')
datfile.visititems(printer)
