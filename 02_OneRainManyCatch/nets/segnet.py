import tensorflow as tf
import numpy as np
import nn

'''
the bug found in the previous version of nn_model was fixed in this file

1. the deconv unit was adjusted to the order that one deconv followed by several convs

2. the channels used in decoding network have been fixed
'''
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
        print("\t",prev)
    
    ls[name] = nn.max_pooling_valid(name, prev, kernel_pool)
    prev = ls[name]
    print("\t",prev)
    return prev

def seg_net_decode_unit(ls, vs, name, x, num, kernel_conv, kernel_pool, input_channel, output_channel, stride, initializer, af):
    '''
    decoding unit, one up sampling (deconv) follow by some conv
    '''
    prev = x
    
    output_shape = [int(x.shape[1]) * kernel_pool[0], int(x.shape[2]) * kernel_pool[1]]
    print("output shape",output_shape)
    cur_name = name + "_dc_"
    ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.deconv_valid(cur_name,prev,kernel_pool,kernel_pool,output_shape,[input_channel,input_channel],initializer,af)
    prev = ls[cur_name]
    print("\t",prev)
    for i in range(num):
        cur_name = name + "_c_" + str(i) if i < num - 1 else name
        # add one layer
        ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.conv_with_pad(cur_name, prev, kernel_conv,stride,[input_channel, output_channel if i == num - 1 else input_channel], initializer, af)
                
        prev = ls[cur_name]
        print("\t",prev)

    return prev

def prediction_network(img_size, input_channel, net_channel, output_channel, kernel_size, pooling_size, initializer, af):
    '''
    this one has bug, it exists only for making the old code working (wrong channel sequence in decoder)
    
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
        ls["x_reshape"] = tf.reshape(phs["x"],[-1,img_size[0],img_size[1],1],"x_reshape")
        prev = ls["x_reshape"]
    
    prev_channel = input_channel
    
    # encoding
    for i in range(len(net_channel)):
        current_channel = net_channel[i]
        print("kernel", kernel_size[i], "in_channel", prev_channel, "out_channel", net_channel[i])
        prev = seg_net_encode_unit(ls, vs,"encode-" + str(i), prev, 2, kernel_size[i], pooling_size[i],prev_channel,current_channel,[1,1],initializer,af)
        prev_channel = current_channel
    
    # decoding
    for i in range(len(net_channel)):
        index = -(i + 1)
        if i < len(net_channel)-1:
            current_channel = net_channel[index-1]
        else:
            current_channel = output_channel
        print("kernel", kernel_size[index], "in_channel", prev_channel, "out_channel", current_channel)
        prev = seg_net_decode_unit(ls, vs,"decode-" + str(len(net_channel)-1-i), prev, 2, kernel_size[index], pooling_size[index], prev_channel, current_channel,[1,1],initializer,af)
        prev_channel = current_channel
        
    if output_channel==1:
        # rank 3 tensor as output
        ls["prediction"] = tf.reshape(prev,[-1,img_size[0],img_size[1]],"prediction")
    else:
        # rank 4 tensor as output
        ls["prediction"] = prev
    return phs, ls, vs