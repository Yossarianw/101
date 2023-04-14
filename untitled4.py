# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 14:39:30 2022

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

im = ps.generators.blobs(shape=[300, 300], porosity=0.18, blobiness=2)

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.imshow(im, origin='lower')
ax.axis(False)

