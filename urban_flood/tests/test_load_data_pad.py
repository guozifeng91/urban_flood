'''
a example file of how to preprocess the data
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def pad_nearest(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = vector[pad_width[0]]
    vector[-pad_width[1]:] = vector[vector.shape[0] - pad_width[1] - 1]
    return vector

path_terrain = "data/luzern/terrain.csv"
path_classes = "data/luzern/classes.csv"

geometry_ = pd.read_csv(path_terrain,header=None)
img_geo = geometry_.values

#img_geo = np.pad(img_geo, (100,200), pad_nearest)


# rescale terrain to -1 to 1
img_geo[img_geo == -9999.0] = np.max(img_geo)
img_geo = ((img_geo - np.min(img_geo)) / (np.max(img_geo) - np.min(img_geo)) - 0.5) * 2

# show elevation data and gradient data
plt.imshow(img_geo,cmap=plt.cm.jet)
plt.show()
