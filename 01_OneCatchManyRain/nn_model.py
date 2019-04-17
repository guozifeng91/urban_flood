import tensorflow as tf
import numpy as np
import nn

# deprecated
IMG_SIZE = 256
# deprecated
PATTERN_RESOLUTION = 12 # the length of the vector that represent a pattern
# deprecated
IMG_CHANNEL = 6 # dem_rescale(0 to 1), mask, slop, cos, sin, curvature 

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
        # print(prev)
    
    ls[name] = nn.max_pooling_valid(name, prev, kernel_pool)
    prev = ls[name]
    # print(prev)
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
        # print(prev)
    
    output_shape = [int(x.shape[1]) * kernel_pool[0], int(x.shape[2]) * kernel_pool[1]]
    # print("output shape",output_shape)

    ls[name],vs[name + "_w"],vs[name + "_b"] = nn.deconv_valid(name,prev,kernel_pool,kernel_pool,output_shape,[input_channel,output_channel],initializer,af)
    prev = ls[name]
    # print(prev)
    return prev

def prediction_network(initializer, af):
    '''
    segnet architecture
    
    deprecated
    '''
    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables

    # 256, terrain, 6 channels (height / mask / slop / aspect cos / aspect sin
    # / curvature)
    phs["x1"] = tf.placeholder(tf.float32,(None,IMG_SIZE,IMG_SIZE,IMG_CHANNEL),name="x1")
    # 12, rain pattern for every 5 minute in 1 hour
    phs["x2"] = tf.placeholder(tf.float32,(None,PATTERN_RESOLUTION),name="x2")

    # build the model
    # encoder
    prev = seg_net_encode_unit(ls, vs,"e1", phs["x1"], 2, [3,3], [2,2],IMG_CHANNEL,16,[1,1],initializer,af)  # 128
    prev = seg_net_encode_unit(ls, vs,"e2", prev, 2, [3,3], [2,2],16,64,[1,1],initializer,af)  # 64
    prev = seg_net_encode_unit(ls, vs,"e3", prev, 2, [3,3], [2,2],64,128,[1,1],initializer,af)  # 32
    
    # rain pattern
    ls["l1"], vs["l1_w"], vs["l1_b"] = nn.linear("l1",phs["x2"],[PATTERN_RESOLUTION,4096],af=af) # 17 -> 4096
    ls["l1_reshape"] = tf.reshape(ls["l1"],[-1,32,32,4],"li_reshape") # 4096 -> 32 * 32 * 4
    ls["concat"] = tf.concat(values=[prev, ls["l1_reshape"]],axis=3,name="concat")
    print(ls["concat"])

    # decoder
    prev = seg_net_decode_unit(ls,vs,"d3", ls["concat"], 2, [3,3], [2,2], 128 + 4, 64,[1,1],initializer,af)
    prev = seg_net_decode_unit(ls,vs,"d2", prev, 2, [3,3], [2,2], 64, 16,[1,1],initializer,af)
    prev = seg_net_decode_unit(ls,vs,"p", prev, 2, [3,3], [2,2], 16, 1,[1,1],initializer,af)

    # 256, prediction of water level
    ls["prediction"] = tf.reshape(prev,[-1,IMG_SIZE,IMG_SIZE],"prediction")
    return phs, ls, vs

def prediction_network_new(img_size, vector_size, linear_size, reshape_size, net_channel, output_channel, kernel_size, pooling_size, initializer, af):
    '''
    segnet architecture
    
    parameters:
    ------------------
    img_size: list of integer, the input image size (height, width) or (height, width, channel)
    vector_size: integer, the length of the input rain vector
    linear_size: list of integer, the linear layers following the rain vector
    reshape_size: list of integer (height, width, channel), reshaped size of the last linear layer
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

    size_all = 1
    for s in reshape_size:
        size_all *= s
        
    if size_all != linear_size[-1]:
        raise Exception("reshape size should match the last item of input size")
        
    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables

    # image(terrain) input x1
    input_channel = 1
    if len(img_size)>2:
        input_channel = img_size[2]
    
    if input_channel != 1:
        # rank 4 tensor as input
        phs["x1"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1],input_channel),name="x1")
        prev = phs["x1"]
    else:
        # rank 3 tensor as input
        phs["x1"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1]),name="x")
        ls["x1_reshape"] = tf.reshape(prev,[-1,img_size[0],img_size[1],1],"x1_reshape")
        prev = ls["x1_reshape"]
    
    prev_channel = input_channel
    
    # encoding of input image x1
    for i in range(len(net_channel)):
        current_channel = net_channel[i]
        prev = seg_net_encode_unit(ls, vs,"encode-" + str(i), prev, 2, kernel_size[i], pooling_size[i],prev_channel,current_channel,[1,1],initializer,af)
        prev_channel = current_channel
        print(prev)

    # vector(rain pattern) input x2
    phs["x2"] = tf.placeholder(tf.float32,(None,vector_size),name="x2")
    prev_linear = phs["x2"]
    prev_lin_size = vector_size
    # linear layers of input vector x2
    for i in range(len(linear_size)):
        name = "l-" + str(i)
        ls[name], vs[name + "_w"], vs[name + "_b"] = nn.linear(name,prev_linear,[prev_lin_size,linear_size[i]],af=af)
        prev_linear = ls[name]
        prev_lin_size = linear_size[i]
        print(prev_linear)
        
    # reshape the last linear layer
    ls["lin_reshape"] = tf.reshape(prev_linear,[-1,reshape_size[0],reshape_size[1],reshape_size[2]],"lin_reshape")
    
    # connect the linear layers with the encoder
    ls["concat"] = tf.concat(values=[prev, ls["lin_reshape"]],axis=3,name="concat")
    print(ls["concat"])
    prev = ls["concat"]
    prev_channel += reshape_size[2] # add the concat channels
    # decoding
    for i in range(len(net_channel)):
        index = -i - 1
        if i < len(net_channel)-1:
            current_channel = net_channel[i]
        else:
            current_channel = output_channel

        prev = seg_net_decode_unit(ls, vs,"decode-" + str(len(net_channel)-1-i), prev, 2, kernel_size[index], pooling_size[index], prev_channel, current_channel,[1,1],initializer,af)
        prev_channel = current_channel
        print(prev)
        
    if output_channel==1:
        # rank 3 tensor as output
        ls["prediction"] = tf.reshape(prev,[-1,img_size[0],img_size[1]],"prediction")
    else:
        # rank 4 tensor as output
        ls["prediction"] = prev
    return phs, ls, vs