import numpy as np

from netCDF4 import Dataset
from matplotlib import pyplot
from mpl_toolkits.basemap import Basemap
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

from CycleGAN_IPSL import *


genA2B = load_model('genA2B.h5')
datasetA, lon, lat = load_A_samples()
datasetB, lon, lat = load_B_samples()
nimage = min(datasetA.shape[0], datasetB.shape[0])
datasetA = datasetA[range(nimage)]
datasetB = datasetB[range(nimage)]

print(datasetA.shape)
for layer in genA2B.layers:
    print(layer.input_shape)
print(genA2B.summary())


# Validation with real pairs
fakesetB = genA2B.predict(datasetA)
nchecks = 4 
datasubA = datasetA[range(nchecks)]
datasubB = datasetB[range(nchecks)]
fakesubB = fakesetB[range(nchecks)]
examples = vstack((datasubA, datasubB, fakesubB, datasubA - datasubB, fakesubB - datasubB))
print(examples.shape)

print('bias genA2B: %f' % np.mean((fakesetB - datasetB)))
print('bias base A: %f' % np.mean((datasetA - datasetB)))
print('mse genA2B: %f' % np.mean((fakesetB - datasetB)**2))
print('mse base A: %f' % np.mean((datasetA - datasetB)**2))
print('corr genA2B: %f' % np.corrcoef(fakesetB.reshape(-1), datasetB.reshape(-1))[0, 1])
print('corr base A: %f' % np.corrcoef(datasetA.reshape(-1), datasetB.reshape(-1))[0, 1])

extent = [-30, 40, 30, 65] # [left, right, bottom, top]
map = Basemap(projection='merc', llcrnrlon=extent[0], urcrnrlon=extent[1], llcrnrlat=extent[2], urcrnrlat=extent[3], resolution='c')
# find x,y of map projection grid.
xx, yy = np.meshgrid(lon, lat)
xx, yy = map(xx, yy)
for i in range(5 * nchecks):
    # define subplot
    pyplot.subplot(5, nchecks, 1 + i)
    # turn off axis
    pyplot.axis('off')
    # plot raw pixel data
    #pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    if (i < (3 * nchecks)): 
        vmin =0
        vmax =1
        cmap = pyplot.get_cmap("YlOrRd")
    else:
        vmin = -0.1
        vmax = 0.1
        cmap = pyplot.get_cmap("RdBu")
    map.pcolormesh(xx,yy, examples[i, :, :, 0], vmin = vmin, vmax = vmax, cmap=cmap)
    map.drawcoastlines(linewidth = 0.2)
    if (i + 1) % nchecks == 0:
        map.colorbar()
    # save plot to file
filename = 'Validation.png'
pyplot.savefig(filename, dpi=150)
pyplot.close()
