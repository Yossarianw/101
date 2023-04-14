# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:43:40 2022

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
tqdm = get_tqdm()
im = ps.generators.blobs(shape=[300, 300], porosity=0.18, blobiness=2)

def ibip(im, inlets=None, dt=None, maxiter=300):
   
    # Process the boundary image
    if inlets is None:
        inlets = get_border(shape=im.shape, mode='faces')
    bd = np.copy(inlets > 0)
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
            inds = np.where(arr)
            result = np.vstack(inds)
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
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=maxiter)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
ibip(im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2))