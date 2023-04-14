# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 07:44:24 2022

@author: 20-2
"""

import numpy as np
from edt import edt
from porespy.tools import get_tqdm
import scipy.ndimage as spim
from porespy.tools import get_border, make_contiguous
from porespy.tools import _insert_disk_at_points
from porespy.tools import Results
import numba
from porespy import settings
import porespy as ps
import matplotlib.pyplot as plt
from libtiff import TIFF
tqdm = get_tqdm()
#im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2)#

import cv2

#img = cv2.imread(r"C:\Users\20-2\Desktop\1111.tif",0)

#print(img)
#print(img.shape)




img = TIFF.open(r'C:\Users\20-2\Desktop\9.14rgb.tif',mode='r')
#tif.write_image(flt, compression=None)
im = img.read_image()
#print(type(im))  # <class 'numpy.ndarray'>
print(im)  # <class 'numpy.uint16'>
#for i, v1 in enumerate(img):
    #for j, v2 in enumerate(v1):
        #if v2 == 255:
            #img[i, j] = False





#print(img)
#print(img.shape)

#plt.imshow(im, cmap='gray')
#im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2)

#print(im)
#print(im.shape)