import tensorflow as tf
import numpy as np
import os
from os.path import join
import math

import matplotlib.pylab as plt
import cv2

# my code
import nn
import nn_model
import lossPloter as lp

import load_data
from load_data import load_waterdepth, load_dem, generate_features, rainfall_pattern

# files
rain_file = "data/744/rain_pattern_str.txt"
dem_file = "data/744/dem/744_dem_asc.asc.gz"
waterdepth_path = "data/744/waterdepth"

model_path = "model"
model_name = "zurich-744_512_01" # nbc refers to normalize by catchment

# parameters
patch_num = 5000
patch_size = 512
input_channel = 6 # dem + mask + slope + cos + sin + curvature
output_channel = 1 # water level
vector_length = 12

lin_size = [4096] # 4096 = 32 * 32 * 4
reshape_size = [32,32,4] # reshape_size is determined by patch_size / pooling_size / .. / pooling_size, which is 32

net_channel = [32,64,128,128]
kernel_size = [[3,3],[3,3],[3,3],[3,3]]
pooling_size = [[2,2],[2,2],[2,2],[2,2]] # size of the latent layer: patch_size / pooling_size / .. / pooling_size = 32

# the cnn model used for training and testing
initializer = tf.contrib.layers.xavier_initializer()
af = tf.nn.leaky_relu
phs, ls, vs = nn_model.prediction_network_new([patch_size, patch_size, input_channel],vector_length, lin_size, reshape_size, net_channel, output_channel, kernel_size, pooling_size, initializer, af)

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
    ls["loss"] = tf.reduce_mean(tf.square(phs["y"] - ls["prediction"]),name="loss")
    train_op = tf.train.AdamOptimizer(0.0002).minimize(ls["loss"])
    loss_op = ls["loss"]
    
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

    # dem_array = np.random.rand(5000, 5000, input_channel) # generate fake data for debug
    dem_array = generate_features(load_dem(dem_file))
   
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
    # loss calculator
    train_loss = lp.Epoch_Loss()
    test_loss = lp.Epoch_Loss()
    ploter = lp.Loss_Ploter()
    
    # recover the loss record
    if last_epoch > 0:
        ploter.load_record(model_path, model_name)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        
        if last_epoch > 0:
            print("restoring")
            saver.restore(sess,join(model_path,model_name+model_id))
        
        print("training")
        
        for e in range(epoch):
            test_loss.clear()
            train_loss.clear()
            
            p = np.copy(patch_train)
            np.random.shuffle(p) # shuffle the training locations (axis 0)
            
            # epoch training
            num_patches = len(patch_train)
            for i in range(0, num_patches, batch_size):
                start = i
                end = min(i+batch_size, num_patches-1)
                # randomly select rainfall patterns for each item in this batch
                rainfall_choice = np.random.choice(ranfall_index_train,end-start)
                # get dem_patches based on the patch locations
                x1 = np.array([dem_array[p[j,0]:p[j,0]+patch_size,p[j,1]:p[j,1]+patch_size] for j in range(start,end)])
                # get rainfall vectors based on the selection of rainfall patterns
                x2 = rainfall_intensity[rainfall_choice]
                # get water level based on the patch locations and the selection of rainfall patterns
                y = np.array([waterdepth_array[p[j,0]:p[j,0]+patch_size,p[j,1]:p[j,1]+patch_size,rainfall_choice[j-start]] for j in range(start,end)])
                
                _,loss_val = sess.run([train_op, loss_op], feed_dict={phs["x1"]:x1,phs["x2"]:x2,phs["y"]:y})
                train_loss.put_batch(loss_val)
                # record batch loss
                ploter.put_batch_loss(loss_val)
                
                # display batch loss curve
                if (ploter.count % 6==5):
                    ploter.plot()
            
            # epoch testing
            p = patch_test # not necessary to shuffle the testing locations
            num_patches = len(patch_test)
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
                
                loss_val = sess.run(loss_op, feed_dict={phs["x1"]:x1,phs["x2"]:x2,phs["y"]:y})
                test_loss.put_batch(loss_val)
            
            # record training and testing loss
            ploter.put_epoch_loss(train_loss.get_epoch_loss(), test_loss.get_epoch_loss())

            print("epoch",e+last_epoch, "train_loss",train_loss.get_epoch_loss(),"test_loss",test_loss.get_epoch_loss())
            ploter.plot()
            ploter.fig.savefig("training_plot.png")
            if e % save_freq == save_freq-1:
                saver.save(sess, join(model_path,model_name+"/"),e + last_epoch)
                ploter.save_record(model_path, model_name)

def run_test(name, dem_array, rain_pattern_vector, target_array=None, random_patch = False, model_id="/-79", batch_size=32, img_height = 20, save_npy=False):
    mask_indice = dem_array < 0
    height, width = dem_array.shape
    
    # necessary data for prediction
    max_x = np.zeros((height,width), dtype=np.float32)
    sum_x = np.zeros((height,width), dtype=np.float32)
    sum_x2 = np.zeros((height,width), dtype=np.float32)
    n_x = np.zeros((height,width), dtype=np.float32)
    stddev = np.zeros((height,width), dtype=np.float32)
    
    # rescale the dem and generate features
    dem_array=generate_features(dem_array)
    
    # generate testing patches
    patches=[]
    num_patches = 0
    if random_patch:
        # generate patch locations
        num_patches = max(batch_size, 3 * math.ceil((height*width)/(patch_size**2)))

        i=0
        while i < num_patches:
            w = np.random.randint(width - patch_size)
            h = np.random.randint(height - patch_size)
            
            if not np.any(dem_array[h:h+patch_size,w:w+patch_size,0]):
                continue
            
            i+=1
            
            patches.append([h,w]) 
    else:
        # generate patch locations
        patches = [[min(h,height-patch_size-1),min(w,width - patch_size-1)] for h in range(0, height-patch_size,patch_size) for w in range(0, width - patch_size, patch_size)]
        # select those that contains catchment areas
        patches = [[h, w] for h, w in patches if np.any(dem_array[h:h+patch_size,w:w+patch_size,0])]
        num_patches = len(patches)
   
    patches = np.array(patches,dtype=np.uint32)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,join(model_path,model_name+model_id))
        print("testing")
        
        for i in range(0, num_patches, batch_size):
            start = i
            end = min(i+batch_size, num_patches-1)
            print(start,end,num_patches)
            
            x1 = np.array([dem_array[patches[j,0]:patches[j,0]+patch_size,patches[j,1]:patches[j,1]+patch_size] for j in range(start,end)])
            x2 = np.array([rain_pattern_vector for _ in range(end-start)])
            
            result = sess.run(ls["prediction"], feed_dict={phs["x1"]:x1,phs["x2"]:x2})
            
            for j in range(end-start):
                h,w = patches[start + j]
                
                max_x[h:h+patch_size,w:w+patch_size] = np.maximum(max_x[h:h+patch_size,w:w+patch_size],result[j])
                sum_x[h:h+patch_size,w:w+patch_size] += result[j]
                sum_x2[h:h+patch_size,w:w+patch_size] += np.square(result[j])
                n_x[h:h+patch_size,w:w+patch_size] += 1
    
    mean_x = np.copy(sum_x)
    mean_x[n_x>0] /= n_x[n_x>0]
    
    # output = max_x
    output = mean_x
    
    if save_npy:
        np.save("test/"+name+"_prediction.npy",output)
        
    # ===================render test results===================
    render_pred = render(output, plt.cm.terrain, 0, 5, mask_indice)
    cv2.imwrite("test/"+name+"_prediction.png",cv2.cvtColor(render_pred,cv2.COLOR_RGBA2BGRA))
    
    render_truth = None
    render_err = None
    
    mask_indice_inv = np.logical_not(mask_indice)
    
    if target_array is not None:
        render_truth = render(target_array, plt.cm.terrain, 0, 5, mask_indice)
        cv2.imwrite("test/"+name+"_truth.png",cv2.cvtColor(render_truth,cv2.COLOR_RGBA2BGRA))
        
        render_err = render(target_array - output, plt.cm.seismic, -3, 3, mask_indice)
        cv2.imwrite("test/"+name+"_error.png",cv2.cvtColor(render_err,cv2.COLOR_RGBA2BGRA))
        
        err = target_array[mask_indice_inv] - output[mask_indice_inv]

        plt.hist(err,50,density=True)
        plt.savefig("test/"+name+"_error_hist.png",bbox_inches="tight")
        plt.close()
        
        plt.hist(err[np.abs(err)>0.1],50,density=True)
        plt.savefig("test/"+name+"_error_hist_0.1.png",bbox_inches="tight")
        plt.close()
        

    mask_indice_inv = n_x > 0
    stddev[mask_indice_inv] = np.sqrt(sum_x2[mask_indice_inv]/n_x[mask_indice_inv] - np.square(sum_x[mask_indice_inv]/n_x[mask_indice_inv]))

    render_stddev = render(stddev, plt.cm.terrain, 0, 1.5, mask_indice)
    cv2.imwrite("test/"+name+"_conf.png",cv2.cvtColor(render_stddev,cv2.COLOR_RGBA2BGRA))
    
    # ===================render test results in one===================
    
    if target_array is None:    
        plt.figure(figsize=(2 * math.ceil(width / height * img_height),img_height))
        plt.subplot(1,2,1)
        plt.gca().set_title("prediction")
        # plt.imshow(output, cmap=plt.cm.terrain,vmin=0,vmax=5)
        plt.imshow(render_pred)
        plt.subplot(1,2,2)
        plt.gca().set_title("stddev(confidence)")
        # plt.imshow(stddev, cmap=plt.cm.jet,vmin=-0.5,vmax=0.1)
        plt.imshow(render_stddev)
        # plt.colorbar()
        plt.savefig("test/"+name+".png",bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(2 * math.ceil(width / height * img_height),2 * img_height))
        plt.subplot(2,2,1)
        plt.gca().set_title("prediction")
        # plt.imshow(output, cmap=plt.cm.terrain,vmin=0,vmax=5)
        plt.imshow(render_pred)
        
        plt.subplot(2,2,3)
        plt.gca().set_title("target")
        # plt.imshow(target_array, cmap=plt.cm.terrain,vmin=0,vmax=5)
        plt.imshow(render_truth)
        
        plt.subplot(2,2,2)
        plt.gca().set_title("stddev(confidence)")
        # plt.imshow(stddev, cmap=plt.cm.jet,vmin=-0.5,vmax=0.1)
        plt.imshow(render_stddev)
        
        plt.subplot(2,2,4)
        plt.gca().set_title("error")
        # plt.imshow(np.abs(target_array - output), cmap=plt.cm.terrain,vmin=0,vmax=5)
        plt.imshow(render_err)
        # plt.colorbar()
        plt.savefig("test/"+name+".png",bbox_inches="tight")
        plt.close()
    return

def render(nd_array, cmap, vmin=0, vmax = 1, mask_indice=None):
    nd_array = (cmap((nd_array - vmin) / (vmax - vmin))*255).astype(np.uint8)
    if mask_indice is not None:
        nd_array[mask_indice] *= 0
    return nd_array

def test(model_id="/-79", random_seed=1, random_patch=True):
    rainfall_name, rainfall_intensity = rainfall_pattern(rain_file)
    # test pattern for zurich-744_256_01:
    # ['tr100-3', 'tr20', 'tr5-3', 'tr10-3', 'tr10', 'tr20-3']
    # [17, 3, 13, 14, 2, 15]

    test_pattern_index = [17, 3, 13, 14, 2, 15]
    
    rainfall_index = np.array([k for k in rainfall_name.keys()],dtype=np.int32)
    np.random.seed(random_seed)
    np.random.shuffle(rainfall_index)
    
    test_pattern_index = rainfall_index[:len(rainfall_index)//3] # 6
    test_pattern_name = [rainfall_name[i] for i in test_pattern_index]
    # for portugal case: [ 3 13  7  2  6] ['tr20-1', 'tr5-3', 'tr5-2', 'tr10-1', 'tr2-2']
    
    print (test_pattern_index, test_pattern_name)
    
    dem_array = load_dem(dem_file)
    waterdepth_array = load_waterdepth(waterdepth_path, test_pattern_name)
    
    for i in range(len(test_pattern_index)):
        run_test(test_pattern_name[i], np.copy(dem_array), rainfall_intensity[test_pattern_index[i]], waterdepth_array[:,:,i],random_patch = random_patch, model_id=model_id)

def test2(dem_file, model_id="/-79"):
    rainfall_name, rainfall_intensity = rainfall_pattern(rain_file)
    test_pattern_index = [17, 3, 13, 14, 2, 15]
    test_pattern_name = [rainfall_name[i] for i in test_pattern_index]
    
    dem_array = load_dem(dem_file)
    # waterdepth_array = load_waterdepth(waterdepth_path, test_pattern_name)
    
    for i in range(len(test_pattern_index)):
        run_test(test_pattern_name[i], np.copy(dem_array), rainfall_intensity[test_pattern_index[i]],model_id=model_id)

# train(32, 80, save_freq = 40)
test(random_patch = False)

# what will happen if on other site?
# test2("D:\\gis data\\zurich\\city catchment\\zurich\\848\\848_dem_asc.asc")

