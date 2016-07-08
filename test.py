import h5py

def prin(item):
    print item

f = h5py.File('tiled_images.hdf5', 'r+')
f.visit(prin)

