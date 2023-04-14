# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 11:20:49 2022

@author: 20-2
"""

import imageio
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
import cv2
from libtiff import TIFF

img = TIFF.open(r'C:\Users\20-2\Desktop\4ct150.tif',mode='r')
im = img.read_image()
#for i, v1 in enumerate(img):
    #for j, v2 in enumerate(v1):
        #if v2 == 255:
            #img[i, j] = True
        
#im = img

tqdm = get_tqdm()   #进度条
#img = imageio.volread(r'C:\Users\20-2\Desktop\rock2.tif')

#im = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
#rendering_img1 = cv2.normalize(img, None, 0, 252, cv2.NORM_MINMAX, cv2.CV_32F)
#im=rendering_img1

inlets = np.zeros_like(im)
inlets[0, ...] = True

def ibip(im, inlets=inlets, dt=None, maxiter=1000):
   
    # Process the boundary image
    #if inlets is None:
        #inlets = get_border(shape=im.shape, mode='faces')
    bd = np.copy(inlets > 0) #深拷贝，拷贝后的地址和拷贝前不一样
    if dt is None:  # Find dt if not given
        dt = edt(im)
    dt = dt.astype(int)  # Conert the dt to nearest integer
    # Initialize inv image with -1 in the solid, and 0's in the void
    inv = -1*(~im)
    sizes = -1*(~im)
    scratch = np.copy(bd)
    for step in tqdm(range(1, maxiter), **settings.tqdm):
        @numba.jit(nopython=True, parallel=False)
        def _where(arr):
            inds = np.where(arr)  #用于找出满足条件的元素位置（坐标）
            result = np.vstack(inds)  #按垂直方向（行顺序）堆叠数组构成一个新的数组
            return result
        pt = _where(bd)
        scratch = np.copy(bd)
        temp = _insert_disk_at_points(im=scratch, coords=pt,
                                       r=1, v=1, smooth=False)
        # Reduce to only the 'new' boundary
        edge = temp*(dt > 0)
        if ~edge.any():
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max = (dt*edge).max()
        # Find all values of the dt with that size
        dt_thresh = dt >= r_max
        # Extract the actual coordinates of the insertion sites
        pt = _where(edge*dt_thresh)
        inv = _insert_disk_at_points(im=inv, coords=pt,
                                      r=r_max, v=step, smooth=True)
        sizes = _insert_disk_at_points(im=sizes, coords=pt,
                                        r=r_max, v=r_max, smooth=True)
        def _update_dt_and_bd(dt, bd, pt):
            if dt.ndim == 2:
                for i in range(pt.shape[1]):
                    bd[pt[0, i], pt[1, i]] = True
                    dt[pt[0, i], pt[1, i]] = 0
            else:
                for i in range(pt.shape[1]):
                    bd[pt[0, i], pt[1, i], pt[2, i]] = True
                    dt[pt[0, i], pt[1, i], pt[2, i]] = 0
            return dt, bd
        dt, bd = _update_dt_and_bd(dt, bd, pt)

    
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=300)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(inv/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=1400)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(inv/im, origin='lower')
    ax.axis(False);
    
ibip(im = img.read_image())
    
    