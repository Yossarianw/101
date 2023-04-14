# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:41:51 2022

@author: 20-2
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

im = io.imread(r'C:\Users\20-2\Desktop\CTdata_00014.bmp')

plt.imshow(im, cmap='gray')

from skimage import restoration
from skimage import img_as_float
im_float = img_as_float(im)

im_denoised = restoration.denoise_nl_means(im_float, h=0.05)

plt.imshow(im_denoised, cmap='gray')
ax = plt.axis('off')

plt.imshow(im_denoised, cmap='gray')
plt.contour(im_denoised, [0.5], colors='yellow')
plt.contour(im_denoised, [0.45], colors='blue')
ax = plt.axis('off')

from skimage import feature
edges = feature.canny(im_denoised, sigma=0.2, low_threshold=0.07, \
                      high_threshold=0.18)
plt.imshow(im_denoised, cmap='gray')
plt.contour(edges)

