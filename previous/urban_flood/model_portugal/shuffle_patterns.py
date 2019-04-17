'''
the latest portugal experiment file
'''

import tensorflow as tf
import numpy as np

# data loading
import load_data.load_npy as ld

# ploting library
import matplotlib.pylab as plt
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as colors

# the cnn model
import nn
import model_portugal.nn_model as nn_model
from model_portugal.nn_model import IMG_SIZE

# name of the rain patterns
SAMPLE_PATTERN = {0:"tr2-1",1:"tr5-1",2:"tr10-1",3:"tr20-1",4:"tr50-1",5:"tr100-1",6:"tr2-2",7:"tr5-2",8:"tr10-2",9:"tr20-2",10:"tr50-2",11:"tr100-2",12:"tr2-3",13:"tr5-3",14:"tr10-3",15:"tr50-3",16:"tr100-3"}

def build_and_train():
    '''
    build and train the cnn model, save the model to the disk

    this function shuffles the rain patterns, and takes 1/4 of them as testing pattern
    '''
    batch_size = 8
    epoche = 500
    patch_num = 600 # (75 batches)

    # load data
    path_terrain = "data/portugal/terrain.npy"
    path_water_level = "data/portugal/water_level.npy"
    path_rain_pattern = "data/portugal/rain_pattern.txt"
    patch_terrain, patch_water_levels, rain_patterns, coords, img_shape = ld.load_terrain(path_terrain, path_water_level, path_rain_pattern, patch_num, IMG_SIZE)

    # shuffle the rain pattern
    np.random.seed(1)
    indice_patterns = np.arange(len(rain_patterns),dtype=np.uint16)
    np.random.shuffle(indice_patterns)

    # the position that seperate the training pattern and test pattern
    divPos = (len(rain_patterns) * 3) // 4

    # seperate the training pattern and testing pattern
    indices_patterns_train = indice_patterns[:divPos]
    indices_patterns_test = indice_patterns[divPos:]

    print("train pattern",indices_patterns_train)
    print("test pattern",indices_patterns_test)

    rain_patterns_train = rain_patterns[indices_patterns_train]
    rain_patterns_test = rain_patterns[indices_patterns_test]

    # seperate water levels by patterns
    train_water_levels = patch_water_levels[:,indices_patterns_train]
    test_water_levels = patch_water_levels[:,indices_patterns_test]

    # ===================build model===================
    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = nn_model.prediction_network(initializer, af)

    # build training network
    phs["y"] = tf.placeholder(tf.float32,(None,IMG_SIZE,IMG_SIZE),name="y")
    ls["loss"] = tf.reduce_mean(tf.square(phs["y"] - ls["prediction"]),name="loss")
    train_op = tf.train.AdamOptimizer(0.0002).minimize(ls["loss"])

    # ===================training===================
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        print("training")

        np.random.seed() # reseed the generator
        for e in range(epoche):
            # two indice lists, one for terran patch, one for rain pattern
            id_patch = np.arange(patch_terrain.shape[0])

            # here the length is counted based on training pattern only
            id_pattern = id_patch % len(rain_patterns_train)

            np.random.shuffle(id_patch)
            np.random.shuffle(id_pattern)

            for i in range(0, patch_num, batch_size):
                # select random patches by id_patch[i:i+batch_size]
                # select random pattern for each patch, by id_pattern[i:i+batch_size]

                # use train water level here
                sess.run(train_op, feed_dict={phs["x1"]:patch_terrain[id_patch[i:i + batch_size]], 

                                              phs["x2"]:rain_patterns_train[id_pattern[i:i + batch_size]], 

                                              phs["y"]:train_water_levels[id_patch[i:i + batch_size],id_pattern[i:i + batch_size]]})
            
            # use test water level here
            print("epoche",e, "loss",get_loss(sess,ls,phs,patch_terrain,test_water_levels, rain_patterns_test))

            if e % 50 == 49:
                saver.save(sess, "data/model/portugal_shuffle_pattern/",e)

def test_model_assemble(path_model):
    '''
    test the trained model using rain pattern from training / testing data
    '''
    errormap = colors.LinearSegmentedColormap.from_list("errormap", plt.cm.Reds(np.linspace(0.15,1,256)), N=256)
    watermap = colors.LinearSegmentedColormap.from_list("watermap", plt.cm.Blues(np.linspace(0.15,1,256)), N=256)

    batch_num = 8

    path_terrain = "data/portugal/terrain.npy"
    path_water_level = "data/portugal/water_level.npy"
    path_rain_pattern = "data/portugal/rain_pattern.txt"

    patch_terrain, patch_water_levels, rain_patterns, grid_shape, img_shape = ld.load_terrain_grid(path_terrain, path_water_level, path_rain_pattern, IMG_SIZE)
    
    np.random.seed(1)
    indice_patterns = np.arange(len(rain_patterns),dtype=np.uint16)
    np.random.shuffle(indice_patterns)
    # the position that seperate the training pattern and test pattern
    divPos = (len(rain_patterns) * 3) // 4
    # seperate the training pattern and testing pattern
    indices_patterns_train = indice_patterns[:divPos]
    indices_patterns_test = indice_patterns[divPos:]

    patch_num = len(patch_terrain)

    predict_water_level = np.zeros([grid_shape[0] * IMG_SIZE,grid_shape[1] * IMG_SIZE])
    truth_water_level = np.zeros([grid_shape[0] * IMG_SIZE,grid_shape[1] * IMG_SIZE])

    # ===================build model===================
    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = nn_model.prediction_network(initializer, af)

    # ===================training===================
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,path_model)

        for pid in range(len(rain_patterns)):
            train_or_test = "train" if np.any(indices_patterns_train==pid) else "test"

            print(SAMPLE_PATTERN[pid], train_or_test)
            i = 0
            while i < patch_num:
                start = i
                end = min(i + batch_num,patch_num)

                # index of patches, from start to end
                index = np.arange(start=start,stop=end)
                # h, w position of patches
                h = index // grid_shape[1]
                w = index % grid_shape[1]
                # rain patther of patches, constant array
                id_pattern = np.ones_like(index,dtype=np.uint8) * pid

                predict = sess.run(ls["prediction"], feed_dict={phs["x1"]:patch_terrain[index], phs["x2"]:rain_patterns[id_pattern]})

                # assemble patches
                for j in range(end - start):
                    truth_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE] = patch_water_levels[index[j],pid]
                    predict_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE] = predict[j]

                    # set mask for the result
                    mask=patch_terrain[index[j],:,:,1]==1
                    truth_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE][mask]=np.nan
                    predict_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE][mask]=np.nan

                i+=batch_num

            # visualizing the result
            error = np.abs(truth_water_level - predict_water_level)
            error_hist=error[np.logical_not(np.isnan(error))].flatten()

            fig = plt.figure(figsize=(40,20),constrained_layout=True)
            gs = GridSpec(2, 4, figure=fig)
            ax1 = fig.add_subplot(gs[0,0])
            ax5 = fig.add_subplot(gs[0,1])

            ax2 = fig.add_subplot(gs[0,2:])
            ax3 = fig.add_subplot(gs[1,2:])
            ax4 = fig.add_subplot(gs[1,0:2])

            ax1.set_ylim(0,160)
            ax1.plot(rain_patterns[pid])
            ax1.set_title(SAMPLE_PATTERN[pid])

            ax2.set_title(SAMPLE_PATTERN[pid] + " truth")
            colorbar(ax2.imshow(truth_water_level,cmap=watermap,vmin=0,vmax=2), cax=make_axes_locatable(ax2).append_axes("right", size="4%", pad="2%"))

            ax3.set_title(SAMPLE_PATTERN[pid] + " prediciton")
            colorbar(ax3.imshow(predict_water_level,cmap=watermap,vmin=0,vmax=2), cax=make_axes_locatable(ax3).append_axes("right", size="4%", pad="2%"))

            ax4.set_title("error")
            colorbar(ax4.imshow(error,cmap=errormap,vmin=0,vmax=0.5), cax=make_axes_locatable(ax4).append_axes("right", size="4%", pad="2%"))

            ax5.set_title("error histogram")
            ax5.hist(error_hist,bins=40,range=(0,0.5),density =True)

            fig.suptitle(train_or_test)

            plt.savefig("data/images/" + train_or_test + " " + SAMPLE_PATTERN[pid] + ".png")

def test_model_assemble_anyrain(path_model, rains=None):
    '''
    test the model with given rain pattern
    '''
    batch_num = 8
    watermap = colors.LinearSegmentedColormap.from_list("watermap", plt.cm.Blues(np.linspace(0.15,1,256)), N=256)

    path_terrain = "data/portugal/terrain.npy"
    path_water_level = "data/portugal/water_level.npy"
    path_rain_pattern = "data/portugal/rain_pattern.txt"

    patch_terrain, patch_water_levels, rain_patterns, grid_shape, img_shape = ld.load_terrain_grid(path_terrain, path_water_level, path_rain_pattern, IMG_SIZE)
    
    if rains is not None:
        rain_patterns = rains

    patch_num = len(patch_terrain)

    predict_water_level = np.zeros([grid_shape[0] * IMG_SIZE,grid_shape[1] * IMG_SIZE])

    # ===================build model===================

    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = nn_model.prediction_network(initializer, af)

    # ===================training===================
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,path_model)

        for pid in range(len(rain_patterns)):
            i = 0
            while i < patch_num:
                start = i
                end = min(i + batch_num,patch_num)

                # index of patches, from start to end
                index = np.arange(start=start,stop=end)
                # h, w position of patches
                h = index // grid_shape[1]
                w = index % grid_shape[1]
                # rain patther of patches, constant array
                id_pattern = np.ones_like(index,dtype=np.uint8) * pid

                predict = sess.run(ls["prediction"], feed_dict={phs["x1"]:patch_terrain[index], phs["x2"]:rain_patterns[id_pattern]})

                # assemble patches
                for j in range(end - start):
                    predict_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE] = predict[j]

                    # set mask for the result
                    mask=patch_terrain[index[j],:,:,1]==1
                    predict_water_level[h[j] * IMG_SIZE:(h[j] + 1) * IMG_SIZE, w[j] * IMG_SIZE:(w[j] + 1) * IMG_SIZE][mask]=np.nan

                i+=batch_num

            # visualization
            fig = plt.figure(figsize=(30,12),constrained_layout=True)
            gs = GridSpec(1, 4, figure=fig)
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1:])

            ax1.set_ylim(0,160)
            ax1.plot(rain_patterns[pid])
            colorbar(ax2.imshow(predict_water_level,cmap=watermap,vmin=0,vmax=2), cax=make_axes_locatable(ax2).append_axes("right", size="4%", pad="2%"))

            plt.savefig("data/images/" + "rnd_" + str(pid) + ".png")


# running start from here:

#build_and_train() # build and train the model
#test_model_assemble("data/model/portugal_shuffle_pattern/-499") # test the model with training and testing patterns

#  test the model with randomly generated patterns
rain_pattern=[]
for i in range(80,160,10):
    for j in range(3):
        rain_pattern.append(np.random.rand(12)*i)

rain_pattern = np.array(rain_pattern)
test_model_assemble_anyrain("data/model/portugal_shuffle_pattern/-499", rain_pattern)