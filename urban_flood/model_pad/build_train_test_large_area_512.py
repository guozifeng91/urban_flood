import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patheffects as path_effects
import nn
import cv2

import load_data.load_csv as load_csv

# values that are used to normalize the heights
HIGHEST_LEVEL = 2000
LOWEST_LEVEL = 0

TERRAIN_SIZE = 512
IMG_CHANNEL = 3

LABEL_SIZE = 256
SIZE_TO = 256

NUM_TYPE = 4

def seg_net_encode_unit(ls, vs, name, x, num, kernel_conv, kernel_pool, input_channel, output_channel, stride, initializer, af):
    '''
    encoding unit, some conv follow by one pooling
    '''
    prev = x

    for i in range(num):
        # name of this module
        cur_name = name + "_c_" + str(i)
        # add one layer
        ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.conv_with_pad(cur_name, prev, kernel_conv,stride,[input_channel, output_channel], initializer, af)
        # change channel
        input_channel = output_channel
                
        prev = ls[cur_name]
        print(prev)
    
    ls[name] = nn.max_pooling_valid(name, prev, kernel_pool)
    prev = ls[name]
    print(prev)
    return prev

def seg_net_decode_unit(ls, vs, name, x, num, kernel_conv, kernel_pool, input_channel, output_channel, stride, initializer, af):
    '''
    decoding unit, some conv follow by one up sampling (deconv)
    '''
    prev = x
    for i in range(num):
        cur_name = name + "_c_" + str(i)
        # add one layer
        ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.conv_with_pad(cur_name, prev, kernel_conv,stride,[input_channel, output_channel], initializer, af)
        # change channel
        input_channel = output_channel
                
        prev = ls[cur_name]
        print(prev)
    
    output_shape = [int(x.shape[1]) * kernel_pool[0], int(x.shape[2]) * kernel_pool[1]]
    print("output shape",output_shape)

    ls[name],vs[name + "_w"],vs[name + "_b"] = nn.deconv_valid(name,prev,kernel_pool,kernel_pool,output_shape,[input_channel,output_channel],initializer,af)
    prev = ls[name]
    print(prev)
    return prev

def prediction_network(initializer, af):
    '''
    segnet architecture
    '''
    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables

    # 3 channels (height / y gradient / x gradient)
    phs["x"] = tf.placeholder(tf.float32,(None,SIZE_TO,SIZE_TO,IMG_CHANNEL),name="x") # 256

    # build the model (unsymmetric)
    prev = seg_net_encode_unit(ls, vs,"e1", phs["x"], 2, [3,3], [2,2],IMG_CHANNEL,16,[1,1],initializer,af)  # 128
    prev = seg_net_encode_unit(ls, vs,"e2", prev, 2, [3,3], [2,2],16,64,[1,1],initializer,af)  # 64
    prev = seg_net_encode_unit(ls, vs,"e3", prev, 2, [3,3], [2,2],64,128,[1,1],initializer,af)  # 32

    prev = seg_net_decode_unit(ls,vs,"d3", prev, 2, [3,3], [2,2], 128, 64,[1,1],initializer,af) # 64
    prev = seg_net_decode_unit(ls,vs,"d2", prev, 2, [3,3], [2,2], 64, 16,[1,1],initializer,af) # 128
    prev = seg_net_decode_unit(ls,vs,"p", prev, 2, [3,3], [2,2], 16, NUM_TYPE,[1,1],initializer,af) #256

    ls["p_index"] = tf.argmax(prev,axis=3,name="p_index")
    print(ls["p_index"])

    return phs, ls, vs

def get_loss(sess, ls, phs, test_x, test_y, batch_size=128):
    indices = np.arange(test_x.shape[0])
    n = 0
    loss_all = 0
    for i in range(0, test_x.shape[0], batch_size):
        loss_val = sess.run(ls["loss"], feed_dict={phs["x"]:test_x[indices[i:i + batch_size]], phs["y"]:test_y[indices[i:i + batch_size]]})
        loss_all += loss_val
        n += 1

    return loss_all / n

def build_and_train():
    batch_size = 16
    epoche = 300

    # ===================load data===================
    # training data
    csv_terrain = "data/luzern/terrain.csv"
    csv_label = "data/luzern/classes.csv"
    train_x, train_y, train_c, train_shape = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,5000,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    # load testing data (different terrain)
    csv_terrain = "data/hoengg/terrain.csv"
    csv_label = "data/hoengg/classes.csv"
    test_x, test_y, _, _ = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,500,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    # a small subset to test
    test_x_2 = train_x[4500:]
    test_y_2 = train_y[4500:]

    # a big subset to train
    train_x = train_x[:4500]
    train_y = train_y[:4500]

    # ===================additional information of the data===================
    # save an image that shows the patches
    test_coord = train_c[4500:]
    train_coord = train_c[:4500]

    # draw rectangles
    patch_img = 255 * np.ones([train_shape[0],train_shape[1],3],dtype=np.uint8)

    for coord in train_coord:
        cv2.rectangle(patch_img,(coord[1],coord[0]),(coord[1] + TERRAIN_SIZE,coord[0] + TERRAIN_SIZE),(255,220,220),thickness=1)
    for coord in test_coord:
        cv2.rectangle(patch_img,(coord[1],coord[0]),(coord[1] + TERRAIN_SIZE,coord[0] + TERRAIN_SIZE),(220,220,255),thickness=1)

    for coord in train_coord:
        patch_img[coord[0] + 510:coord[0] + 515,coord[1] + 510:coord[1] + 515,1:3] = 0
    for coord in test_coord:
        patch_img[coord[0] + 510:coord[0] + 515,coord[1] + 510:coord[1] + 515,0:2] = 0

    cv2.imwrite("data/images/patches.png",patch_img)

    # ===================build model===================
    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = prediction_network(initializer, af)

    # build training network
    phs["y"] = tf.placeholder(tf.int32,(None,SIZE_TO,SIZE_TO),name="y")
    ls["y_onehot"] = tf.one_hot(phs["y"],NUM_TYPE,1.0,0.0,-1,dtype=tf.float32,name="y_onehot")
    print(ls["y_onehot"])

    ls["loss"] = tf.reduce_mean(tf.square(ls["y_onehot"] - ls["p"]),name="loss")
    train_op = tf.train.AdamOptimizer(0.0002).minimize(ls["loss"])

    # ===================training===================
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()

        print("training")

        for e in range(epoche):
            indices = np.arange(train_x.shape[0])
            np.random.shuffle(indices)
            for i in range(0, train_x.shape[0], batch_size):
                sess.run(train_op, feed_dict={phs["x"]:train_x[indices[i:i + batch_size]], phs["y"]:train_y[indices[i:i + batch_size]]})

            print("epoche",e, "loss subset",get_loss(sess,ls,phs,test_x_2,test_y_2), "loss hoengg",get_loss(sess,ls,phs,test_x,test_y))

            if e % 50 == 49:
                saver.save(sess, "data/model/seg_net_model_pad_512/",e)

def test_model(path):
    batch_size = 10
    sample = 500
    # load training data
    csv_terrain = "data/luzern/terrain.csv"
    csv_label = "data/luzern/classes.csv"
    train_x, train_y, _, _ = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,sample,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    # load testing data (different terrain)
    csv_terrain = "data/hoengg/terrain.csv"
    csv_label = "data/hoengg/classes.csv"
    test_x, test_y, _, _ = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,sample,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    #initializer = tf.truncated_normal_initializer(0, 0.01)
    af = tf.nn.leaky_relu
    phs, ls, vs = prediction_network(initializer, af)

    # values for visualization
    min_x_train = np.min(train_x[:,:,:,0])
    max_x_train = np.max(train_x[:,:,:,0])

    min_x_test = np.min(test_x[:,:,:,0])
    max_x_test = np.max(test_x[:,:,:,0])

    print(min_x_train,max_x_train)
    print(min_x_test,max_x_test)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,path)

        for i in range(10):
            print("round",i)
            indices = np.arange(sample)
            np.random.shuffle(indices)

            # show test result on luzern
            x = train_x[indices[0:batch_size]]
            y = train_y[indices[0:batch_size]]
            p = sess.run(ls["p_index"],feed_dict={phs["x"]:x})
            plot_img(x,y,p,batch_size,min_x_train,max_x_train,"data/images/luzern_result_" + str(i) + ".png")

            # show test result on hoengg
            x = test_x[indices[0:batch_size]]
            y = test_y[indices[0:batch_size]]
            p = sess.run(ls["p_index"],feed_dict={phs["x"]:x})
            plot_img(x,y,p,batch_size,min_x_test,max_x_test,"data/images/hoengg_result_" + str(i) + ".png")

def test_model_assemble(path):
    batch_size = 128

    # load training data
    csv_terrain = "data/luzern/terrain.csv"
    csv_label = "data/luzern/classes.csv"
    train_x, train_y, train_c, train_shape = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,5000,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    train_x = train_x[4500:]
    train_y = train_y[4500:]
    train_c = train_c[4500:]

    # load testing data (different terrain)
    #csv_terrain = "data/hoengg/terrain.csv"
    #csv_label = "data/hoengg/classes.csv"
    #train_x, train_y, train_c, train_shape = load_csv.load_expand_terrain_scale(csv_terrain,csv_label,500,TERRAIN_SIZE,LABEL_SIZE,SIZE_TO,dtype_img=np.float16)

    offset = (TERRAIN_SIZE - LABEL_SIZE) // 2

    # propability prediction of patches
    train_p = np.zeros(np.concatenate((train_y.shape,[NUM_TYPE])))
    # ground truth of propability prediction
    train_y_onehot = np.zeros(np.concatenate((train_y.shape,[NUM_TYPE])))

    print(train_y.shape,train_p.shape)

    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = prediction_network(initializer, af)

    phs["y"] = tf.placeholder(tf.int32,(None,SIZE_TO,SIZE_TO),name="y")
    ls["y_onehot"] = tf.one_hot(phs["y"],NUM_TYPE,1.0,0.0,-1,dtype=tf.float32,name="y_onehot")

    # run prediction
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,path)

        start = 0
        while start < 500:
            end = min(start + batch_size, 500)
            print(start,"to",end)

            # get prediction probability
            train_p[start:end] = sess.run(ls["p"],feed_dict={phs["x"]:train_x[start:end]})
            train_y_onehot[start:end] = sess.run(ls["y_onehot"],feed_dict={phs["y"]:train_y[start:end]})

            start += batch_size

    assemble(train_shape, train_p, train_c, LABEL_SIZE, offset, "prediction")
    assemble(train_shape, train_y_onehot, train_c, LABEL_SIZE, offset, "truth")

def assemble(shape, prediction, patches_coord, label_size, offset, name):
    '''
    assemble patches back to the full size input image.
    
    inputs
    ------
    shape: shape of the full size images
    prediction: probablistic prediction (has size of label_size)
    patches_coord: coordinates of the patches at left top corner
    label_size: size of the label
    offset: (max(terrain_size, label_size) - label_size) // 2
    name: name of the saved image
    '''
    img_ass_prob = np.zeros((shape[0],shape[1],NUM_TYPE)) - 1
    # assemble patches, the probabilities in the overlapping area would be the
    # maximum value
    for i in range(len(prediction)):
        img_ass_prob[patches_coord[i,0] + offset:patches_coord[i,0] + offset + label_size,patches_coord[i,1] + offset:patches_coord[i,1] + offset + label_size] = \
        np.maximum(img_ass_prob[patches_coord[i,0] + offset:patches_coord[i,0] + offset + label_size,patches_coord[i,1] + offset:patches_coord[i,1] + offset + label_size],prediction[i])

    # plot probability distribution
    plt.figure(figsize=(NUM_TYPE * 10,10))
    for i in range(NUM_TYPE):
        plt.subplot(1,NUM_TYPE,i + 1)
        plt.imshow(img_ass_prob[:,:,i][::-1].astype(np.float32),cmap=plt.cm.jet)

    plt.savefig("data/images/prob_" + name + ".png")

    # plot types
    # plus 1 to keep class 0 in visualization
    img_ass = np.argmax(img_ass_prob,axis=2) + 1
    # mask of the non-prediction area
    img_mask = np.all(img_ass_prob == -1,axis=2)
    # set the area within the mask to class 0
    img_ass[img_mask] = 0

    plt.figure(figsize=(10,10))
    plt.imshow(img_ass[::-1].astype(np.float32),cmap=plt.cm.jet,vmin=0,vmax=NUM_TYPE)
    plt.savefig("data/images/label_" + name + ".png")

def plot_img(x,y,p,batch_size,vmin,vmax,filename=None):
    '''
    plot 3 channel image, the ground truth and the prediction
    '''
    plt.figure(figsize=(batch_size / 5 * 10,10))
    
    line_size = batch_size + 1
    for i in range(batch_size):
        # the image
        plt.subplot(5,line_size,i + 2)
        plt.imshow(x[i,:,:,0].astype(np.float32),cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
        plt.subplot(5,line_size,i + line_size + 2)
        plt.imshow(x[i,:,:,1].astype(np.float32),cmap=plt.cm.jet,vmin=-1,vmax=1)
        plt.subplot(5,line_size,i + 2 * line_size + 2)
        plt.imshow(x[i,:,:,2].astype(np.float32),cmap=plt.cm.jet,vmin=-1,vmax=1)

        # the label
        # plus 1 to keep class 0 in visualization
        plt.subplot(5,line_size,i + 3 * line_size + 2)
        plt.imshow(1 + y[i],cmap=plt.cm.jet,vmin=0,vmax=NUM_TYPE)
        plt.subplot(5,line_size,i + 4 * line_size + 2)
        plt.imshow(1 + p[i],cmap=plt.cm.jet,vmin=0,vmax=NUM_TYPE)

    plt.subplot(5,line_size, 1)
    plt.text(0,0.5,"height")
    plt.axis('off')

    plt.subplot(5,line_size, line_size + 1)
    plt.text(0,0.5,"gradient y")
    plt.axis('off')

    plt.subplot(5,line_size, 2 * line_size + 1)
    plt.text(0,0.5,"gradient x")
    plt.axis('off')

    plt.subplot(5,line_size, 3 * line_size + 1)
    plt.text(0,0.5,"label")
    plt.axis('off')

    plt.subplot(5,line_size, 4 * line_size + 1)
    plt.text(0,0.5,"prediction")
    plt.axis('off')

    #plt.show()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

#build_and_train()
#test_model("data/model/seg_net_model_pad_512/-299")
#test_model_confusion_matrix("data/model/seg_net_model_2/-499")
test_model_assemble("data/model/seg_net_model_pad_512/-299")