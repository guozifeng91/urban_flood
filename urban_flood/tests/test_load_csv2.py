import load_data.load_csv as load_csv
import matplotlib.pylab as plt
import numpy as np

csv_terrain = "data/luzern/terrain.csv"
csv_label = "data/luzern/classes.csv"

train_x, train_y, _, _ = load_csv.load_terrain_scale(csv_terrain, csv_label, 10, 1024, 256, 100, dtype_img=np.float32, dtype_label=np.uint8, seed=10)

for i in range(10):
    plt.subplot(3,10,i + 1)
    plt.imshow(train_x[i,:,:,0],cmap=plt.cm.jet,vmin=-1,vmax=1)
    
    plt.subplot(3,10,i + 1 + 10)
    plt.imshow(train_x[i,:,:,1],cmap=plt.cm.jet,vmin=-1,vmax=1)

    plt.subplot(3,10,i + 1 + 20)
    plt.imshow(train_x[i,:,:,2],cmap=plt.cm.jet,vmin=-1,vmax=1)
   
plt.show()

train_x, train_y, train_c, shape = load_csv.load_terrain_scale(csv_terrain, csv_label, 10000, 64, 64, 50, dtype_img=np.float16, dtype_label=np.uint8, seed=1)
print(shape)
img = np.zeros(shape,dtype=np.uint8)
for i in range(len(train_x)):
    img[train_c[i][0] - 10:train_c[i][0] + 10,train_c[i][1] - 10:train_c[i][1] + 10] = train_y[i] + 1

plt.imshow(img,cmap=plt.cm.jet)
plt.show()