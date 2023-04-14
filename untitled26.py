# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:42:37 2023

@author: 20-2
"""

import numpy as np
import porespy.filters as pf
from skimage import measure, io
import matplotlib.pyplot as plt

# 读取指定的 TIFF 格式的图片
im = io.imread(r'C:\Users\20-2\Desktop\CT160.tif')
# 对图像进行分割
regions = pf.local_thickness(im, sizes=range(1, 10, 2))
# 强制将标签转换为整数类型
regions = regions.astype(int)
# 计算孔隙的尺寸分布
props = measure.regionprops_table(regions, properties=['label', 'area'])
hist, bin_edges = np.histogram(props['area'], bins=100)

# 显示生成的图像
io.imshow(im)
io.show()