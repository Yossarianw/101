# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:39:32 2022

@author: 20-2
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from skimage import io

im = io.imread(r'C:\Users\20-2\Desktop\files\CTdata_00113.bmp')

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
x, y = np.random.random((2, 1000))
xk, yk = trim_close_points((x, y), 0.1)
plt.plot(x, y, 'o')
plt.plot(xk, yk, 'or')

from time import time
from sklearn.neighbors import kneighbors_graph

#import vertex_coloring 

from skimage import morphology,segmentation,measure
from skimage.segmentation import random_walker

from pyamg.graph import vertex_coloring
n_real = 1000
n_markers = 5000
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
                                       beta=25000, mode='cg_mg')
    segmentations.append(labels_rw)
t2 = time()
#print(t2 - t1)

segmentations = np.array(segmentations)
boundaries = np.zeros_like(im_denoised[::2, ::2])
for seg in segmentations:
    boundaries += segmentation.find_boundaries(seg, connectivity=2)
    
plt.imshow(boundaries, cmap='gist_heat'); plt.colorbar()    

#plt.savefig(r"C:\Users\20-2\Desktop\rock00.tif", dpi=600)
#plt.show()

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


def color_segmentation(regions, n_neighbors=25):
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
 
                                                beta=25000, mode='cg_mg')

#plt.savefig(r"C:\Users\20-2\Desktop\rock0.tif", dpi=600)
#plt.show()



plt.imshow(color.label2rgb(final_segmentation, im_denoised[::2, ::2],
                    colors=plt.cm.Spectral(np.linspace(0, 1, 40))))
ax = plt.axis('off')

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.savefig(r"C:\Users\20-2\Desktop\Rock113a.tif", dpi=600)
plt.show()

plt.imshow(np.random.permutation(final_segmentation.max() + 1)
                                    [final_segmentation], 
                                    cmap='Spectral')

ax = plt.gca()
ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)

plt.savefig(r"C:\Users\20-2\Desktop\save\CT113.tif", dpi=600)
plt.show()





