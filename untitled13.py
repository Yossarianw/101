# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:28:00 2022

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
tqdm = get_tqdm()
im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2)



plt.imshow(im, cmap='gist_heat')

ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)



plt.savefig(r"C:\Users\20-2\Desktop\Rockc.tif", dpi=600)
plt.show()