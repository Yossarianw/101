# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:35:32 2023

@author: 20-2
"""
import porespy.filters as pf
import porespy.metrics as pm
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.measure import regionprops
from scipy.stats import rankdata
import porespy.visualization as vis
import matplotlib.cm as cm

import os
print(os.environ.get('PYTHONPATH'))

img = plt.imread(r'C:\Users\20-2\Desktop\CT160.tif')

# 打印图片的形状和数据类型
print('Image shape:', img.shape)
print('Image dtype:', img.dtype)

# 打印图片数据的最大值、最小值、平均值等统计信息
print('Image max:', np.max(img))
print('Image min:', np.min(img))
print('Image mean:', np.mean(img))

# 读取图像
im = io.imread(r'C:\Users\20-2\Desktop\CT160.tif')
io.imshow(im)
# 对图像进行分割
regions = pf.local_thickness(im, sizes=range(1, 5000, 1))
# 强制将标签转换为整数类型
regions = regions.astype(int)
# 计算孔隙的尺寸分布
props = regionprops(regions)
# 将孔隙按大小排序
sizes = np.array([prop.area for prop in props])
sizes_seq = rankdata(sizes)
# 可视化排序后的孔洞
plt.figure(figsize=[8, 8])



plt.imshow(regions, cmap='gray')
plt.show()
