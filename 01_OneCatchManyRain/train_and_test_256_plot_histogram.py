import tensorflow as tf
import numpy as np
import os
from os.path import join
import math

import matplotlib.pylab as plt
import cv2

# my code
import nn
import nn_model_new # the one that fixed the bug in nn_model
import lossPloter as lp

import load_data
from load_data import load_waterdepth, load_dem, rainfall_pattern

#dem_process_func = load_data.generate_features
dem_process_func = load_data.rescale_dem_negative_positive

# files
rain_file = "data/luzern/rain_pattern_str.txt"
dem_file = "data/luzern/dem/luzern.asc.gz"
waterdepth_path = "data/luzern/waterdepth"

model_path = "model2"

#model_name = "luzern_256_features"
model_name = "luzern_256_neg_pos"

test_path = "test-" + model_path + "-" + model_name
if not os.path.exists(test_path):
    os.mkdir(test_path)
    
# parameters
patch_num = 10000
patch_size = 256

#input_channel = load_data.channels_generate_features
input_channel = load_data.channels_rescale_dem_negative_positive

output_channel = 1 # water level
vector_length = 12

lin_size = [4096] # 4096 = 32 * 32 * 4
reshape_size = [32,32,4] # reshape_size is determined by patch_size / pooling_size / .. / pooling_size, which is 32

net_channel = [32,64,128]
kernel_size = [[3,3],[3,3],[3,3]]
pooling_size = [[2,2],[2,2],[2,2]] # size of the latent layer: patch_size / pooling_size / .. / pooling_size = 32

# the cnn model used for training and testing
initializer = tf.contrib.layers.xavier_initializer()
af = tf.nn.leaky_relu
phs, ls, vs = nn_model_new.prediction_network([patch_size, patch_size, input_channel],vector_length, lin_size, reshape_size, net_channel, output_channel, kernel_size, pooling_size, initializer, af)

def train(batch_size, epoch, save_freq = 40, random_seed = 1, last_epoch=0):
    '''
    build and train the cnn model, save the model to the disk
    
    parameters:
    ----------------
    random_seed: random seed for patch generation
    last_epoch: load the pretrained model and continue training, None by default
    '''
    model_id = "/-" + str(last_epoch-1)
    
    # ===================build training model===================
    phs["y"] = tf.placeholder(tf.float32,(None,patch_size,patch_size),name="y")
    ls["error"] = phs["y"] - ls["prediction"]
    error_op = ls["error"]
    
    # ===================load and generate training data===================
    np.random.seed(random_seed)
    
    # rain fall pattern
    rainfall_name, rainfall_intensity = rainfall_pattern(rain_file)
    assert rainfall_intensity.shape[1]==vector_length
    rainfall_index = np.array([k for k in rainfall_name.keys()],dtype=np.int32)
      
    np.random.shuffle(rainfall_index)
    ranfall_index_test = rainfall_index[:len(rainfall_index)//3] # 6
    ranfall_index_train = rainfall_index[len(rainfall_index)//3:] # 12
    
    print("test pattern",ranfall_index_test, [rainfall_name[i] for i in ranfall_index_test])

    # for portugal: test pattern [ 3 13  7  2  6] ['tr20-1', 'tr5-3', 'tr5-2', 'tr10-1', 'tr2-2']
    # for zurich and luzern: test pattern [ 6  3 13  2 14  7] ['tr2-2', 'tr20', 'tr5-3', 'tr10', 'tr10-3', 'tr5-2']
    
    # dem_array = np.random.rand(5000, 5000, input_channel) # generate fake data for debug
    dem_array = dem_process_func(load_dem(dem_file))
   
    height, width, _ = dem_array.shape
    
    # waterdepth_array = np.random.rand(height, width, len(rainfall_name)) # generate fake data for debug
    waterdepth_array = load_waterdepth(waterdepth_path, rainfall_name.values()) # for training
    
    patches = [] # locatrion of patches
    
    i=0
    while i < patch_num:
        w = np.random.randint(width - patch_size)
        h = np.random.randint(height - patch_size)
        
        if not np.any(dem_array[h:h+patch_size,w:w+patch_size,0]):
            continue
        
        i+=1
        
        patches.append([h,w])
        
    patches = np.array(patches,dtype=np.uint32)
    patch_test = patches[:patch_num//10]
    patch_train = patches[patch_num//10:]
    
    # ===================training===================
    all_error = np.array([])
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        
        if last_epoch > 0:
            print("restoring")
            saver.restore(sess,join(model_path,model_name+model_id))
        
        print("training")
        
        for e in range(epoch):
            # epoch testing
            p = patch_test # not necessary to shuffle the testing locations
            num_patches = len(p)
            for i in range(0, num_patches, batch_size):
                start = i
                end = min(i+batch_size, num_patches-1)
                # randomly select rainfall patterns for each item in this batch
                rainfall_choice = np.random.choice(ranfall_index_test,end-start)
                # get dem_patches based on the patch locations
                x1 = np.array([dem_array[p[j,0]:p[j,0]+patch_size,p[j,1]:p[j,1]+patch_size] for j in range(start,end)])
                # get rainfall vectors based on the selection of rainfall patterns
                x2 = rainfall_intensity[rainfall_choice]
                # get water level based on the patch locations and the selection of rainfall patterns
                y = np.array([waterdepth_array[p[j,0]:p[j,0]+patch_size,p[j,1]:p[j,1]+patch_size,rainfall_choice[j-start]] for j in range(start,end)])
                
                error_val = sess.run([error_op], feed_dict={phs["x1"]:x1,phs["x2"]:x2,phs["y"]:y})
                all_error=np.concatenate((all_error, error_val), axis=None)
            
            # record training and testing loss
            plt.figure(figsize=(10,5))
            plt.hist(all_error, bins=[0.2*(i-40/2) for i in range(41)], color = "gray", rwidth=0.85)
            plt.yscale("log")
            plt.ylim((1, len(all_error)))
            plt.xlabel('Prediction Error',fontsize=16)
            plt.ylabel('Frequency',fontsize=16)
            plt.suptitle("Histogram of " + ("Multi-channel" if model_name.split("_")[-1]=="features" else "Single-channel") + " Model", fontsize=20)
            plt.savefig('for_paper/' + model_name + '_hist.png', dpi=300)
            plt.close()
            break

train(32, 80, save_freq = 40, last_epoch=80)
