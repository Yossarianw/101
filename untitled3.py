# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:05:07 2022

@author: 20-2
"""

import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
ps.visualization.set_mpl_style()

import inspect
inspect.signature(ps.filters.snow_partitioning)

img =  r"C:\Users\20-2\Desktop\CTdata_00011.bmp"
im=plt.imread(img)
snow = ps.filters.snow_partitioning(im)

fig, ax = plt.subplots(1, 2, figsize=[12, 6])
ax[0].imshow(snow.dt/im/~snow.peaks, origin='lower', interpolation='none')
ax[0].axis(False)
ax[1].imshow(snow.regions/im, origin='lower', interpolation='none')
ax[1].axis(False);

np.random.seed(4)
img =  r"C:\Users\20-2\Desktop\CTdata_00011.bmp"
im=plt.imread(img)
snow1 = ps.filters.snow_partitioning(im, sigma=0)
snow2 = ps.filters.snow_partitioning(im, sigma=0.5)
snow3 = ps.filters.snow_partitioning(im, sigma=1.5)

fig, ax = plt.subplots(1, 3, figsize=[15, 5])
ax[0].imshow(snow1.dt/im/~snow1.peaks, origin='lower', interpolation='none')
ax[0].axis(False)
ax[1].imshow(snow2.dt/im/~snow2.peaks, origin='lower', interpolation='none')
ax[1].axis(False)
ax[2].imshow(snow3.dt/im/~snow3.peaks, origin='lower', interpolation='none')
ax[2].axis(False);

np.random.seed(4)
img =  r"C:\Users\20-2\Desktop\CTdata_00011.bmp"
im=plt.imread(img)
snow1 = ps.filters.snow_partitioning(im, r_max=1)
snow2 = ps.filters.snow_partitioning(im, r_max=3)
snow3 = ps.filters.snow_partitioning(im, r_max=5)

fig, ax = plt.subplots(1, 3, figsize=[15, 5])
ax[0].imshow(snow1.dt/im/~snow1.peaks, origin='lower', interpolation='none')
ax[0].axis(False)
ax[1].imshow(snow2.dt/im/~snow2.peaks, origin='lower', interpolation='none')
ax[1].axis(False)
ax[2].imshow(snow3.dt/im/~snow3.peaks, origin='lower', interpolation='none')
ax[2].axis(False);

