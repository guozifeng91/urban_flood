import tensorflow as tf
import numpy as np
import nn

IMG_SIZE = 256
PATTERN_RESOLUTION = 12 # the length of the vector that represent a pattern
IMG_CHANNEL = 6

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

def get_loss(sess, ls, phs, test_terrain, test_water_level, rain_patterns, batch_size=100):
    indices = np.arange(test_terrain.shape[0])
    n = 0
    loss_all = 0

    # for each pattern
    for j in range(len(rain_patterns)):
        id_pattern = np.ones(batch_size,dtype=np.uint16) * j
        # for each patch
        for i in range(0, test_terrain.shape[0], batch_size):
            loss_val = sess.run(ls["loss"], feed_dict={phs["x1"]:test_terrain[indices[i:i + batch_size]], phs["x2"]:rain_patterns[id_pattern], phs["y"]:test_water_level[indices[i:i + batch_size],id_pattern]})
            loss_all += loss_val
            n += 1

    return loss_all / n

#initializer = tf.contrib.layers.xavier_initializer()
#af = tf.nn.leaky_relu
#phs, ls, vs = prediction_network(initializer, af)