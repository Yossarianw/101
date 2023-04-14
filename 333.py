# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 10:18:34 2022

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

from scipy import spatial

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
x, y = np.random.random((2, 50))
xk, yk = trim_close_points((x, y), 0.1)
plt.plot(x, y, 'o')
plt.plot(xk, yk, 'or')

from time import time
from sklearn.neighbors import kneighbors_graph

#import vertex_coloring 

from skimage import morphology,segmentation
from skimage.segmentation import random_walker

from pyamg.graph import vertex_coloring
n_real = 400
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
                                       beta=25000, mode='cg_mg')
    segmentations.append(labels_rw)
t2 = time()
#print(t2 - t1)

segmentations = np.array(segmentations)
boundaries = np.zeros_like(im_denoised[::2, ::2])
for seg in segmentations:
    boundaries += segmentation.find_boundaries(seg, connectivity=2)
    
plt.imshow(boundaries, cmap='gist_heat'); plt.colorbar()   




import cv2
path = boundaries     #图片路径
#img = cv.imdecode(np.fromfile("动漫人物_0.jpg",np.uint8))#含有中文路径的图片打开
img = cv2.imread(path)  #读取图片
cv2.imwrite(r"C:\Users\20-2\Desktop\rock.jpg",img) 
 