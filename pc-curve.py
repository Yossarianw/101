# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:23:25 2022

@author: 20-2
"""

import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF
np.random.seed(0)

img = TIFF.open(r'C:\Users\20-2\Desktop\5ct150.tif',mode='r')
im = img.read_image()

#im = ps.generators.blobs(shape=[200, 200], porosity=0.181)
drainage = ps.filters.porosimetry(im)
#data = ps.metrics.pc_curve(im=im, sizes=drainage, voxel_size=1e-5)
#plt.plot( data.pc,data.snwp,'b-o')
#plt.xlabel('Capillary Pressure [Pa]')
#plt.ylabel('Non-wetting Phase Saturation');

#fig, ax = plt.subplots(1, 2, figsize=[12, 6])
#ax[0].step(data.pc, data.snwp, 'b-o', where='post')
#ax[0].set_xlabel('Capillary Pressure [Pa]')
#ax[0].set_ylabel('Non-wetting Phase Saturation')
#ax[1].semilogx(data.pc, data.snwp, 'r-o')
#ax[1].set_xlabel('Capillary Pressure [Pa]')
#ax[1].set_ylabel('Non-wetting Phase Saturation');

sigma = 0.072
theta = 180
voxel_size = 1e-5
pc = -2*sigma*np.cos(np.deg2rad(theta))/(drainage*voxel_size)

data = ps.metrics.pc_curve(im=im, pc=pc)
plt.plot(data.pc, data.snwp, 'b-o')
plt.xlabel('Capillary Pressure [Pa]')
plt.ylabel('Non-wetting Phase Saturation');



