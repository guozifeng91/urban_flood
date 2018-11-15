'''
a example file of how to preprocess the data
'''

import numpy as np
import pandas as pd
import matplotlib.pylab as plt


path_terrain = "data/hoengg/terrain.csv"
path_classes = "data/hoengg/classes.csv"

classes_ = pd.read_csv(path_classes,header=None)
img_cls = classes_.values

geometry_ = pd.read_csv(path_terrain,header=None)
img_geo = geometry_.values

# a mask that detect if the data patch out of boundary
mask = (img_geo < 0)

# get gradient, clip between -1 and 1
img_gradient = np.gradient(img_geo,1,1)
img_gradient[0][img_gradient[0] >= 1] = 1
img_gradient[1][img_gradient[1] >= 1] = 1
img_gradient[0][img_gradient[0] <= -1] = -1
img_gradient[1][img_gradient[1] <= -1] = -1

#img_gradient = np.gradient(img_geo,100,100)
#img_gradient[0][img_gradient[0] >= 0.01] = 0.01
#img_gradient[1][img_gradient[1] >= 0.01] = 0.01
#img_gradient[0][img_gradient[0] <= -0.01] = -0.01
#img_gradient[1][img_gradient[1] <= -0.01] = -0.01

# rescale terrain to -1 to 1
img_geo[img_geo == -9999.0] = np.max(img_geo)
img_geo = ((img_geo - np.min(img_geo)) / (np.max(img_geo) - np.min(img_geo)) - 0.5) * 2

# show elevation data and gradient data
fig = plt.figure(figsize=(18,10))
plt.subplot(1,4,1)
plt.imshow(img_geo[::-1],cmap=plt.cm.jet)
plt.subplot(1,4,2)
plt.imshow(img_gradient[0][::-1],cmap=plt.cm.jet)
plt.subplot(1,4,3)
plt.imshow(img_gradient[1][::-1],cmap=plt.cm.jet)
plt.subplot(1,4,4)
plt.imshow(img_cls[::-1],cmap=plt.cm.jet,vmin=0,vmax=4)

plt.show()

#fig = plt.figure(figsize=(10,10))
#plt.imshow(img_cls[::-1],cmap=plt.cm.jet,vmin=0,vmax=4)
#plt.savefig("data/images/label.png")