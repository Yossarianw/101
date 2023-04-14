# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:03:53 2022

@author: 20-2
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

im = io.imread(r'C:\Users\20-2\Desktop\CTdata_00014.tif')

plt.imshow(im, cmap='gray')

from skimage import restoration
from skimage import img_as_float
im_float = img_as_float(im)

im_denoised = restoration.denoise_nl_means(im_float, h=0.05)

from scipy import spatial,ndimage

def trim_close_points(points, distance=1):
    """
    Greedy method to remove some points so that
    all points are separated by a distance greater
    than ``distance``.
    
    points : array of shape (2, n_points)
        Coordinates of 2-D points
        
    distance : float
        Minimal distance between points
    """
    x, y = points
    tree = spatial.KDTree(np.array([x, y]).T)
    pairs = tree.query_pairs(distance)
    remove_indices = []
    for pair in pairs:
        if pair[0] in remove_indices:
            continue
        if pair[1] in remove_indices:
            continue
        else:
            remove_indices.append(pair[1])
    keep_indices = np.setdiff1d(np.arange(len(x)), remove_indices)
    return np.array([x[keep_indices], y[keep_indices]])

# Check result on simple example
x, y = np.random.random((2, 10))
xk, yk = trim_close_points((x, y), 0.1)
plt.plot(x, y, 'o')
plt.plot(xk, yk, 'or')

from time import time
from sklearn.neighbors import kneighbors_graph

#import vertex_coloring 

from skimage import morphology,segmentation,measure
from skimage.segmentation import random_walker

from pyamg.graph import vertex_coloring
n_real = 100
n_markers = 50
segmentations = []
t1 = time()
for real in range(n_real):
    # Random markers
    x, y = np.random.random((2, n_markers))
    x *= im_denoised.shape[0]
    y *= im_denoised.shape[1]
    # Remove points too close to each other
    xk, yk = trim_close_points((x, y), 20)
    mat = kneighbors_graph(np.array([xk, yk]).T, 12)
    
    colors = vertex_coloring(mat)
    # Array of markers
    markers_rw = np.zeros(im_denoised.shape, dtype=np.int)
    markers_rw[xk.astype(np.int), yk.astype(np.int)] = colors + 1
    markers_rw = morphology.dilation(markers_rw, morphology.disk(3))
    # Segmentation
    labels_rw = segmentation.random_walker(im_denoised[::2, ::2], 
                                           markers_rw[::2, ::2],\
                                       beta=2500, mode='cg_mg')
    segmentations.append(labels_rw)
t2 = time()
#print(t2 - t1)

segmentations = np.array(segmentations)
boundaries = np.zeros_like(im_denoised[::2, ::2])
for seg in segmentations:
    boundaries += segmentation.find_boundaries(seg, connectivity=2)
    
plt.imshow(boundaries, cmap='gist_heat'); plt.colorbar()    

def hysteresis_thresholding(im, v_low, v_high):
    """
    Parameters
    ----------
    im : 2-D array
    
    v_low : float
        low threshold
        
    v_high : float
        high threshold
    """
    mask_low = im > v_low
    mask_high = im > v_high
    # Connected components of mask_low
    labels_low = measure.label(mask_low, background=0) + 1
    count = labels_low.max()
    # Check if connected components contain pixels from mask_high
    sums = ndimage.sum(mask_high, labels_low, np.arange(count + 1))
    good_label = np.zeros((count + 1,), bool)
    good_label[1:] = sums[1:] > 0
    output_mask = good_label[labels_low]
    return output_mask   

from skimage import measure, color


def color_segmentation(regions, n_neighbors=2):
    """
    Reduce the number of labels in a label image to make
    visualization easier.
    """
    count = regions.max()
    centers = ndimage.center_of_mass(regions + 2, regions, 
                                     index=np.arange(1, count + 1))
    centers = np.array(centers)
    mat = kneighbors_graph(np.array(centers), n_neighbors)
    colors = vertex_coloring(mat)
    colors = np.concatenate(([0], colors))
    return colors[regions]                       


def plot_colors(val_low, val_high):
    """
    Plot result of segmentation superimposed on original image,
    and plot original image as well.
    """
    seg = hysteresis_thresholding(boundaries, val_low, val_high)
    regions = measure.label(np.logical_not(seg),
                            background=0, connectivity=1)
    color_regions = color_segmentation(regions)
    colors = [plt.cm.Spectral(val) for val in 
                   np.linspace(0, 1, color_regions.max() + 1)]
    image_label_overlay = color.label2rgb(color_regions, 
                                          im_denoised[::2, ::2],
                                          colors=colors)
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(image_label_overlay)
    plt.subplot(122)
    plt.imshow(im_denoised, cmap='gray')
    return regions


regions=plot_colors(0.3 * n_real, 0.55 * n_real)
regions_clean = morphology.remove_small_objects(regions + 1, min_size=60)
regions_clean, _, _ = segmentation.relabel_sequential(regions_clean)
plt.imshow(np.random.permutation(regions_clean.max() + 1)[regions_clean],
                   cmap='Spectral')

final_segmentation = segmentation.random_walker(im_denoised[::2, ::2], 
                                                regions_clean,
 
                                                beta=2500, mode='cg_mg')
#import cv2
#path = final_segmentation     #图片路径
#img = cv.imdecode(np.fromfile("动漫人物_0.jpg",np.uint8))#含有中文路径的图片打开
#img = cv2.imread(path)  #读取图片
#cv2.imwrite(r"C:\Users\20-2\Desktop\rock.jpg",img)  #将图片保存为1.jpg

plt.imshow(color.label2rgb(final_segmentation, im_denoised[::2, ::2],
                    colors=plt.cm.Spectral(np.linspace(0, 1, 40))))
ax = plt.axis('off')

plt.imshow(np.random.permutation(final_segmentation.max() + 1)
                                    [final_segmentation], 
                                    cmap='Spectral')


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
im = plt.imshow(np.random.permutation(final_segmentation.max() + 1)
                                    [final_segmentation], 
                                    cmap='Spectral')

inlets = np.zeros_like(im)
inlets[0, ...] = True

def ibip(im, inlets=inlets, dt=None, maxiter=100):
   
    # Process the boundary image
    #if inlets is None:
        #inlets = get_border(shape=im.shape, mode='faces')
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
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=100)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=150)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=200)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=250)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=300)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
    
    ip = ps.filters.ibip(im=im, inlets=inlets, maxiter=1000)

    fig, ax = plt.subplots(1, 1, figsize=[6, 6])
    ax.imshow(ip.inv_sequence/im, origin='lower')
    ax.axis(False);
ibip(im = plt.imshow(np.random.permutation(final_segmentation.max() + 1)
                                    [final_segmentation], 
                                    cmap='Spectral'))
plt.savefig(?600dpi)