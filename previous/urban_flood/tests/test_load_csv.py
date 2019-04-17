import load_data.load_csv as load_csv
import matplotlib.pylab as plt

csv_terrain = "data/luzern/terrain.csv"
csv_label = "data/luzern/classes.csv"

train_x, train_y, _, _ = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,5,512,512,512)
test_x, test_y, _, _ = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,5,512,256,512)

for i in range(10):
    plt.subplot(2,10,i+1)
    if i < 5:
        plt.imshow(train_x[i,:,:,0],cmap=plt.cm.jet)
    else:
        plt.imshow(test_x[i-5,:,:,0],cmap=plt.cm.jet)

    plt.subplot(2,10,i+1+10)

    if i < 5:
        plt.imshow(train_y[i],cmap=plt.cm.jet)
    else:
        plt.imshow(test_y[i-5],cmap=plt.cm.jet)
plt.show()