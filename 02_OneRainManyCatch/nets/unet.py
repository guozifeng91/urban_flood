import tensorflow as tf
import numpy as np
import nn

'''
unet is the segnet that the feature maps of the encoder part are copied to the decoder part 
'''
def encode_unit(ls, vs, name, x, num, kernel_conv, kernel_pool, conv_channel, stride, initializer, af):
    '''
    encoding unit, some conv follow by one pooling (optional)
    '''
    prev = x
    
    in_channel = prev.shape[-1]

    for i in range(num):
        # name of this module
        cur_name = name + "-" + str(i)
        # add one layer
        ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.conv_with_pad(cur_name, prev, kernel_conv,stride,[in_channel, conv_channel], initializer, af)
        # change channel
        in_channel = conv_channel
                
        prev = ls[cur_name]
        print("\t",prev)
    
    if kernel_pool is not None:
        ls[name] = nn.max_pooling_valid(name + "-p", prev, kernel_pool)
        prev = ls[name]
        print("\t",prev)
        
    return prev

def decode_unit(ls, vs, name, x, x_skip, num, kernel_conv, kernel_pool, conv_channel, stride, initializer, af, output_channel = None):
    '''
    decoding unit, one up sampling (conv transpose) follow by some conv
    
    output_channel: None by default, specify the output channel of the last conv layer
    '''
    prev = x
    
    if output_channel is None:
        output_channel = conv_channel
    
    output_shape = [int(x.shape[1]) * kernel_pool[0], int(x.shape[2]) * kernel_pool[1]]
    print("output shape",output_shape)
    
    x_channel = x.shape[-1]
    x_skip_channel = x_skip.shape[-1]
    
    cur_name = name + "-dc"
    ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.deconv_valid(cur_name,prev,kernel_pool,kernel_pool,output_shape,[x_channel,conv_channel],initializer,af)
    prev = ls[cur_name]
    print("\t",prev)
    
    cur_name = name + "-conc"
    prev = tf.concat([x_skip, prev],axis=-1,name=cur_name)
    ls[cur_name] = prev
    print("\t",prev)
    
    for i in range(num):
        cur_name = name + "-" + str(i) #if i < num - 1 else name
        # add one layer
        ls[cur_name],vs[cur_name + "_w"],vs[cur_name + "_b"] = nn.conv_with_pad(cur_name, prev, kernel_conv,stride,[(conv_channel + x_skip_channel) if i == 0 else conv_channel, output_channel if i == num - 1 else conv_channel], initializer, af)
                
        prev = ls[cur_name]
        print("\t",prev)

    return prev

def unet_sym(img_size, input_channel, encode_channel, output_channel, encode_kernel_size, pooling_size, initializer, af):
    return unet(img_size, input_channel, encode_channel, None, None, output_channel, encode_kernel_size, None, None, pooling_size, initializer, af)
    
def unet(img_size, input_channel, encode_channel, latent_channel, decode_channel, output_channel, encode_kernel_size, latent_kernel_size, decode_kernel_size, pooling_size, initializer, af):
    '''
    parameters:
    ------------------
    img_size: list of integer, the input image size (height, width)
    input_channel: integer, channel of the input image
    encode_channel: list of integer, channels of the encoding network
    latent_channel: integer, channel of the latent network, if none is passed, the lase encode_channel will be used
    decode_channel: list of integer, channels of the decoding network, if none is passed, the reversed encode_channel will be used
    output_channel: integer, channel of the output image
    encode_kernel_size: list of integer, kernel sizes of the encoding network
    latent_kernel_size: integer, kernel size of the latent network, if none is passed, the last encod_kernel_size will be used
    decode_kernel_size: list of integer, kernel sizes of the decoding network, if none is passed, the reversed encode_kernel_size will be used
    pooling_size: list of integer, pooling kernel sizes of the encoding network (decoding network will use the reversed list)
    initializer: variable initializer
    af: activation function
    '''
    phs = {} # placeholders
    ls = {} # layers
    vs = {} # variables
    
    if decode_channel is None:
        decode_channel = list(reversed(encode_channel))
    
    if decode_kernel_size is None:
        decode_kernel_size = list(reversed(encode_kernel_size))
    
    if latent_kernel_size is None:
        latent_kernel_size = encode_kernel_size[-1]
    
    if latent_channel is None:
        latent_channel = encode_channel[-1]
    
    if input_channel != 1:
        # rank 4 tensor as input
        phs["x"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1],input_channel),name="x")
        prev = phs["x"]
    else:
        # rank 3 tensor as input
        phs["x"] = tf.placeholder(tf.float32,(None,img_size[0],img_size[1]),name="x")
        ls["x_reshape"] = tf.reshape(phs["x"],[-1,img_size[0],img_size[1],1],"x_reshape")
        prev = ls["x_reshape"]

    # encoding
    for i in range(len(encode_channel)):
        out_channel = encode_channel[i]
        print("kernel", encode_kernel_size[i], "out_channel", encode_channel[i])
        prev = encode_unit(ls, vs,"encode" + str(i), prev, 2, encode_kernel_size[i], pooling_size[i],encode_channel[i],[1,1],initializer,af)
    
    # latent conv_unit (conv layers without pooling)
    # do something here
    prev = encode_unit(ls, vs,"latent", prev, 2, latent_kernel_size, None, latent_channel,[1,1],initializer,af)
    
    # decoding
    for i in range(len(decode_channel)):
        index = len(decode_channel)-(i + 1) # pooling size use the reversed list of encoder
        
        #name = "concat"+str(i)
        #prev = tf.concat([ls["encode" + str(index) + "-1"], prev],axis=-1,name=name)
        #ls[name] = prev
        
        # encode_channel + previous decode_channel is the input channel number
        in_channel = prev.shape[-1]
        # in_channel = encode_channel[index] + (XXX if i == 0 else decode_channel[i-1])
        
        print("kernel", decode_kernel_size[i], "out_channel", out_channel)
        
        if i < len(decode_channel)-1:
            prev = decode_unit(ls, vs,"decode" + str(i), prev, ls["encode" + str(index) + "-1"], 2, decode_kernel_size[i], pooling_size[index], decode_channel[i],[1,1],initializer,af)
        else:
            prev = decode_unit(ls, vs,"decode" + str(i), prev, ls["encode" + str(index) + "-1"], 2, decode_kernel_size[i], pooling_size[index], decode_channel[i],[1,1],initializer,af, output_channel = output_channel)
        # in_channel = out_channel
        
    if output_channel==1:
        # rank 3 tensor as output
        ls["prediction"] = tf.reshape(prev,[-1,img_size[0],img_size[1]],"prediction")
    else:
        # rank 4 tensor as output
        ls["prediction"] = prev
    return phs, ls, vs