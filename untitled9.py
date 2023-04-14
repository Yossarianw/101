# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 15:02:20 2022

@author: 20-2
"""

import numpy as np
import openpnm as op
import porespy as ps
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)
np.random.seed(10)
#%matplotlib inline
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)

resolution = 2.25e-6
name = 'Bentheimer'

# Read input RAW file
raw_file = np.fromfile("C:\Users\20-2\Desktop\新建\ct图片\CTdata_00014.bmp", dtype=np.float)  
im = (raw_file.reshape(1000,1000))
im = im==0;


#NBVAL_IGNORE_OUTPUT
fig, ax = plt.subplots(1, 1, figsize=(12,5))
ax[0].imshow(im[:, 100]);

ax[0].set_title("Slice No. 100 View");

