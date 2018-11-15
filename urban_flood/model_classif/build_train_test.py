import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patheffects as path_effects
import nn
import cv2
import load_data.load_csv as load_csv

DELTA_HEIGHT_MAX = 50

IMG_SIZE = 256
IMG_CHANNEL = 3
NUM_TYPE = 4

def prediction_network(initializer, af):
    '''
    segnet architecture
    '''
    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables

    # 3 channels (height / y gradient / x gradient)
    phs["x"] = tf.placeholder(tf.float32,(None,IMG_SIZE,IMG_SIZE,IMG_CHANNEL),name="x") # 256

    ls["c1"], vs["c1_w"], vs["c1_b"] = nn.conv_valid("c1",phs["x"],[5,5],[1,1],[IMG_CHANNEL,16],initializer,af) # 252
    ls["p1"] = nn.max_pooling_valid("p1",ls["c1"],[2,2]) #126

    ls["c2"], vs["c2_w"], vs["c2_b"] = nn.conv_valid("c2",ls["p1"],[3,3],[1,1],[16,32],initializer,af) # 124
    ls["p2"] = nn.max_pooling_valid("p2",ls["c2"],[2,2]) #62

    ls["c3"], vs["c3_w"], vs["c3_b"] = nn.conv_valid("c3",ls["p2"],[3,3],[1,1],[32,64],initializer,af) # 60
    ls["p3"] = nn.max_pooling_valid("p3",ls["c3"],[2,2]) #30

    ls["c4"], vs["c4_w"], vs["c4_b"] = nn.conv_valid("c4",ls["p3"],[3,3],[1,1],[64,128],initializer,af) # 28
    ls["p4"] = nn.max_pooling_valid("p4",ls["c4"],[2,2]) #14

    ls["c5"], vs["c5_w"], vs["c5_b"] = nn.conv_valid("c5",ls["p4"],[3,3],[1,1],[128,256],initializer,af) # 12
    ls["p5"] = nn.max_pooling_valid("p5",ls["c5"],[2,2]) #6

    ls["c6"], vs["c6_w"], vs["c6_b"] = nn.conv_valid("c6",ls["p5"],[3,3],[1,1],[256,512],initializer,af) # 4
    ls["c7"], vs["c7_w"], vs["c7_b"] = nn.conv_valid("c7",ls["c6"],[3,3],[1,1],[512,512],initializer,af) # 2
    ls["p7"] = nn.max_pooling_valid("p7",ls["c7"],[2,2]) #1

    ls["p7_reshape"] = tf.reshape(ls["p7"], [-1, 512],name="p7_reshape") #512
    ls["l8"], vs["l8_w"], vs["l8_b"] = nn.linear("l8",ls["p7_reshape"],[512, 1024],af=af)

    # last layer, use sigmoid
    ls["p"], vs["p_w"], vs["p_b"] = nn.linear("p",ls["l8"],[1024, NUM_TYPE],af=af)

    ls["p_index"] = tf.argmax(ls["p"],axis=1,name="p_index")
    for i in ls:
        print(ls[i])

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
    batch_size = 8
    epoche = 300

    # ===================load data===================
    # training data, reduce the samples to 10000
    csv_terrain = "data/luzern/terrain.csv"
    csv_label = "data/luzern/classes.csv"
    train_x, train_y, train_c, train_shape = load_csv.load_terrain_scale(csv_terrain, csv_label, 5000, 512, 256, DELTA_HEIGHT_MAX, dtype_img=np.float16, dtype_label=np.uint8, seed=1)

    # load testing data (different terrain)
    csv_terrain = "data/hoengg/terrain.csv"
    csv_label = "data/hoengg/classes.csv"
    test_x, test_y, _, _ = load_csv.load_terrain_scale(csv_terrain, csv_label, 500, 512, 256, DELTA_HEIGHT_MAX, dtype_img=np.float16, dtype_label=np.uint8, seed=1)

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
        cv2.rectangle(patch_img,(coord[1] - 256,coord[0] - 256),(coord[1] + 256,coord[0] + 256),(255,220,220),thickness=1)
    for coord in test_coord:
        cv2.rectangle(patch_img,(coord[1] - 256,coord[0] - 256),(coord[1] + 256,coord[0] + 256),(220,220,255),thickness=1)

    for coord in train_coord:
        patch_img[coord[0] - 5:coord[0] + 5,coord[1] - 5:coord[1] + 5,1:3] = 0
    for coord in test_coord:
        patch_img[coord[0] - 5:coord[0] + 5,coord[1] - 5:coord[1] + 5,0:2] = 0

    cv2.imwrite("data/images/patches.png",patch_img)

    # ===================build model===================
    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = prediction_network(initializer, af)

    # build training network
    phs["y"] = tf.placeholder(tf.int32,(None),name="y")
    ls["y_onehot"] = tf.one_hot(phs["y"],NUM_TYPE,1.0,0.0,-1,dtype=tf.float32,name="y_onehot")
    print(ls["y_onehot"])

    ls["loss"] = tf.reduce_mean(tf.square(ls["y_onehot"] - ls["p"]),name="loss")
    train_op = tf.train.AdamOptimizer(0.0001).minimize(ls["loss"])

    lb_truth = tf.placeholder(tf.int32,(None),name="label_truth")
    lb_pred = tf.placeholder(tf.int32,(None),name="label_pred")
    conf_matrix = tf.confusion_matrix(lb_truth,lb_pred,NUM_TYPE)

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

            if e % 10 == 9:
                # comput confusion matrix every 2 epoches
                lb_pred_all = np.zeros(500,dtype=np.uint8)

                # test set
                for i in range(25):
                    lb_pred_all[i * 20:(i + 1) * 20] = sess.run(ls["p_index"], feed_dict={phs["x"]:test_x_2[i * 20:(i + 1) * 20]})

                print("confusion matrix subset")
                print(sess.run(conf_matrix, feed_dict={lb_truth:test_y_2, lb_pred:lb_pred_all}))
                
                # hoengg
                for i in range(25):
                    lb_pred_all[i * 20:(i + 1) * 20] = sess.run(ls["p_index"], feed_dict={phs["x"]:test_x[i * 20:(i + 1) * 20]})

                print("confusion matrix hoengg")
                print(sess.run(conf_matrix, feed_dict={lb_truth:test_y, lb_pred:lb_pred_all}))


            if e % 50 == 49:
                # save model every 50 epoches
                saver.save(sess, "data/model/seg_net_model_classif/",e)

def test_model_assemble(path):
    batch_size = 128

    # load training data
    csv_terrain = "data/luzern/terrain.csv"
    csv_label = "data/luzern/classes.csv"
    train_x, train_y, train_c, train_shape = load_csv.load_terrain_scale(csv_terrain, csv_label, 5000, 512, 256, DELTA_HEIGHT_MAX, dtype_img=np.float16, dtype_label=np.uint8, seed=1)

    #csv_terrain = "data/hoengg/terrain.csv"
    #csv_label = "data/hoengg/classes.csv"
    #train_x, train_y, train_c, train_shape =
    #load_data(csv_terrain,csv_label,1000,256)

    # propability prediction of patches
    train_p = np.zeros(np.concatenate((train_y.shape,[NUM_TYPE])))
    print(train_y.shape,train_p.shape)

    # build prediction network
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    phs, ls, vs = prediction_network(initializer, af)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess,path)

        start = 0
        while start < 1000:
            end = min(start + batch_size, 1000)
            print(start,"to",end)

            # get prediction probability
            train_p[start:end] = sess.run(ls["p"],feed_dict={phs["x"]:train_x[start:end]})

            start += batch_size

    img_ass_prob = np.zeros((train_shape[0],train_shape[1],NUM_TYPE)) - 1
    # assemble patches, the probabilities in the overlapping area would be the
    # maximum value
    for i in range(5000):
        img_ass_prob[train_c[i,0]-5:train_c[i,0]+5,train_c[i,1]-5:train_c[i,1]+5] = np.maximum(img_ass_prob[train_c[i,0]-5:train_c[i,0]+5,train_c[i,1]-5:train_c[i,1]+5
                                                                                                            ],train_p[i])

    # plot probability distribution
    plt.figure(figsize=(NUM_TYPE * 10,10))
    for i in range(NUM_TYPE):
        plt.subplot(1,NUM_TYPE,i + 1)
        plt.imshow(img_ass_prob[:,:,i][::-1].astype(np.float32),cmap=plt.cm.jet)
    plt.savefig("data/images/prob.png")

    # plot types
    # plus 1 to keep class 0 in visualization
    img_ass = np.argmax(img_ass_prob,axis=2) + 1
    # mask of the non-prediction area
    img_mask = np.all(img_ass_prob == -1,axis=2)
    # set the area within the mask to class 0
    img_ass[img_mask] = 0

    plt.figure(figsize=(10,10))
    plt.imshow(img_ass[::-1].astype(np.float32),cmap=plt.cm.jet,vmin=0,vmax=NUM_TYPE)
    plt.savefig("data/images/prediction.png")


def plot_img(x,y,p,batch_size,vmin,vmax,filename=None):
    '''
    plot 3 channel image, the ground truth and the prediction
    '''
    plt.figure(figsize=(batch_size / 5 * 10,10))
    
    line_size = batch_size + 1
    for i in range(batch_size):
        # the image
        plt.subplot(5,line_size,i + 2)
        plt.imshow(x[i,:,:,0],cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
        plt.subplot(5,line_size,i + line_size + 2)
        plt.imshow(x[i,:,:,1],cmap=plt.cm.jet,vmin=-1,vmax=1)
        plt.subplot(5,line_size,i + 2 * line_size + 2)
        plt.imshow(x[i,:,:,2],cmap=plt.cm.jet,vmin=-1,vmax=1)

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

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

#build_and_train()
test_model_assemble("data/model/seg_net_model_classif/-299")