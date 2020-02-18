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

nloc = datasetA.shape[1] * datasetA.shape[2]
iloc = range(0, nloc, nloc // nchecks)
for i in iloc:
    print(i, end=', ')
print('\n')
dataqqA = datasetA.reshape((-1, nloc, 1))
dataqqB = datasetB.reshape((-1, nloc, 1))
fakeqqB = fakesetB.reshape((-1, nloc, 1))
fig, axes = pyplot.subplots(nrows = 2, ncols = nchecks, sharex = True, sharey = True)
for i in range(nchecks):
    print('%d, %d' % (1 + i, nchecks + i + 1))
    # Quantile-quantile plot
    axes[0, i].scatter(np.sort(dataqqA[:, iloc[i], 0]), np.sort(dataqqB[:, iloc[i], 0]))
    axes[0, i].plot([0,1], [0, 1], 'r-', lw=2)
    axes[1, i].scatter(np.sort(fakeqqB[:, iloc[i], 0]), np.sort(dataqqB[:, iloc[i], 0]))
    axes[1, i].plot([0,1], [0, 1], 'r-', lw=2)
    #pyplot.xlabel('X')
    #pyplot.ylabel('Y')
filename = 'QQplot.png'
pyplot.savefig(filename, dpi=150)
pyplot.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

fig, axes = pyplot.subplots(nrows = 1, ncols = 3, sharex = True, sharey = True)

frame = 0
axes[0].set_title("A")
map.pcolormesh(xx,yy, datasetA[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[0])
map.drawcoastlines(linewidth = 0.2, ax=axes[0])
axes[1].set_title("B")
map.pcolormesh(xx,yy, datasetB[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[1])
map.drawcoastlines(linewidth = 0.2, ax=axes[1])
axes[2].set_title("genA2B(A)")
map.pcolormesh(xx,yy, fakesetB[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[2])
map.drawcoastlines(linewidth = 0.2, ax=axes[2])
#pyplot.show()

def update(frame):
    axes[0].set_title("A")
    ln1 = map.pcolormesh(xx,yy, datasetA[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[0])
    map.drawcoastlines(linewidth = 0.2, ax=axes[0])
    axes[1].set_title("B")
    ln2 = map.pcolormesh(xx,yy, datasetB[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[1])
    map.drawcoastlines(linewidth = 0.2, ax=axes[1])
    axes[2].set_title("genA2B(A)")
    ln3 = map.pcolormesh(xx,yy, fakesetB[frame, :, :, 0], vmin = 0, vmax = 1, cmap=cmap, ax=axes[2])
    map.drawcoastlines(linewidth = 0.2, ax=axes[2])
    return [ln1, ln2, ln3]

ani = FuncAnimation(fig, update, frames=range(100), blit=False)
#pyplot.show()
writer = animation.writers['ffmpeg'](fps=30)
ani.save('demo.mp4',writer=writer,dpi=150)
