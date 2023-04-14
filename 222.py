# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import porespy as ps
from loguru import logger
ps.visualization.set_mpl_style()

import inspect
inspect.signature(ps.filters.ibip)

img1 =  r"C:\Users\20-2\Desktop\CTdata_00014.bmp"
im=plt.imread(img1)
img = 255 * np.array(im.astype("uint8"))
im=img
ip = ps.filters.ibip(im=im)

fig, ax = plt.subplots(1, 2, figsize=[12, 6])
ax[0].imshow(ip.inv_sequence/im, origin='lower')
ax[0].axis(False)
ax[1].imshow(ip.inv_sizes/im, origin='lower')
ax[1].axis(False);