import tensorflow as tf
import numpy as np
import nn

'''
huge bugs have found in 2019-05-16 from this file:
1. the deconv unit has the wrong order of layers
2. the decoding network has the wrong order of channels,

this file is deprecated now, it exists only for making the old code working
'''
IMG_SIZE = 256
FEATURE_LEVEL = 6 # change here every time new patches are generated
FEATURE_NUM = 4
INPUT_CHANNEL = FEATURE_LEVEL * FEATURE_NUM + 1
OUTPUT_CHANNEL = 1

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

    # 256, terrain, 6 channels (height / mask / slop / aspect cos / aspect sin
    # / curvature)
    phs["x1"] = tf.placeholder(tf.float32,(None,IMG_SIZE,IMG_SIZE,INPUT_CHANNEL),name="x1")

    # build the model
    # encoder
    prev = seg_net_encode_unit(ls, vs,"e1", phs["x1"], 2, [3,3], [2,2],INPUT_CHANNEL,32,[1,1],initializer,af)  # 128
    prev = seg_net_encode_unit(ls, vs,"e2", prev, 2, [3,3], [2,2],32,64,[1,1],initializer,af)  # 64
    prev = seg_net_encode_unit(ls, vs,"e3", prev, 2, [3,3], [2,2],64,128,[1,1],initializer,af)  # 32

    # decoder
    prev = seg_net_decode_unit(ls,vs,"d3", prev, 2, [3,3], [2,2], 128, 64,[1,1],initializer,af)
    prev = seg_net_decode_unit(ls,vs,"d2", prev, 2, [3,3], [2,2], 64, 16,[1,1],initializer,af)
    prev = seg_net_decode_unit(ls,vs,"p", prev, 2, [3,3], [2,2], 16, OUTPUT_CHANNEL,[1,1],initializer,af)

    # 256, prediction of water level
    ls["prediction"] = tf.reshape(prev,[-1,IMG_SIZE,IMG_SIZE],"prediction")
    return phs, ls, vs

def prediction_network_new(img_size, input_channel, net_channel, output_channel, kernel_size, pooling_size, initializer, af):
    '''
    segnet architecture
    
    parameters:
    ------------------
    img_size: list of integer, the input image size (height, width)
    input_channel: integer, channel of the input image
    net_channel: list of integer, channels of the encoding network (decoding network will use the reversed list)
    output_channel: integer, channel of the output image
    kernel_size: list of integer, kernel sizes of the encoding network (decoding network will use the reversed list)
    pooling_size: list of integer, pooling kernel sizes of the encoding network (decoding network will use the reversed list)
    initializer: variable initializer
    af: activation function
    '''
    if len(net_channel) != len(kernel_size):
        raise Exception("kernel_size should have the same length with net_channel")
    if len(pooling_size) != len(net_channel):
        raise Exception("pooling_size should have the same length with net_channel")

    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables

    if input_channel != 1:
        # rank 4 tensor as input
        phs["x"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1],input_channel),name="x")
        prev = phs["x"]
    else:
        # rank 3 tensor as input
        phs["x"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1]),name="x")
        ls["x_reshape"] = tf.reshape(prev,[-1,img_size[0],img_size[1],1],"x_reshape")
        prev = ls["x_reshape"]
    
    prev_channel = input_channel
    
    # encoding
    for i in range(len(net_channel)):
        current_channel = net_channel[i]
        prev = seg_net_encode_unit(ls, vs,"encode-" + str(i), prev, 2, kernel_size[i], pooling_size[i],prev_channel,current_channel,[1,1],initializer,af)
        prev_channel = current_channel
    
    # decoding
    for i in range(len(net_channel)):
        index = -i - 1
        if i < len(net_channel)-1:
            current_channel = net_channel[i]
        else:
            current_channel = output_channel
        
        prev = seg_net_decode_unit(ls, vs,"decode-" + str(len(net_channel)-1-i), prev, 2, kernel_size[index], pooling_size[index], prev_channel, current_channel,[1,1],initializer,af)
        prev_channel = current_channel
        
    if output_channel==1:
        # rank 3 tensor as output
        ls["prediction"] = tf.reshape(prev,[-1,img_size[0],img_size[1]],"prediction")
    else:
        # rank 4 tensor as output
        ls["prediction"] = prev
    return phs, ls, vs