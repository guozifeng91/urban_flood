import tensorflow as tf
import numpy as np

# in case using a older version of tensorflow and no leaky relu is implemented
def lrelu(x, alpha):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def max_pooling_valid(name, x, kernel_shape):
    return tf.nn.max_pool(x, [1, kernel_shape[0], kernel_shape[1], 1], [1, kernel_shape[0], kernel_shape[1], 1], padding='VALID', name=name)

def conv_valid(name, x, kernel_shape, stride, channel, initializer, af):
    '''
    add conv2d layer with VALID padding method:
    
    output_size = (input_size - kernel_size) / stride + 1
    '''
    w = tf.get_variable(name + "_w", shape=[kernel_shape[0], kernel_shape[1], channel[0], channel[1]], initializer=initializer)
    b = tf.get_variable(name + "_b", shape=(channel[1]), initializer=tf.constant_initializer(0.0))
    return af(tf.nn.conv2d(x, w, strides=(1, stride[0], stride[1], 1), padding='VALID') + b, name=name), w, b

def conv_with_pad(name, x, kernel_shape, stride, channel, initializer, af):
    '''
    this method first pad the input then make conv on it, to make the result shape equals to the input shape
    '''
    
    # x: [batch, h, w, channel]
    h = x.shape[1]
    w = x.shape[2]
   
    # the remainder of stride
    r_h = (h - kernel_shape[0]) % stride[0]
    r_w = (w - kernel_shape[1]) % stride[1]
    
    # number of stride (output dimension of VALID padding)
    d_h = (h - kernel_shape[0]) // stride[0] + 1
    d_w = (w - kernel_shape[1]) // stride[1] + 1
    
    # if remainder is not 0, extend output stride number
    if (r_h > 0):
        d_h = d_h + 1
        
    if (r_w > 0):
        d_w = d_w + 1
    
    # the stride number needed to make input and output have the same size
    d_h = h - d_h
    d_w = w - d_w
        
    # extend the remainder (padding needed)
    r_h += d_h * stride[0]
    r_w += d_w * stride[1]
     
    p_w_left = r_w // 2
    p_w_right = r_w - p_w_left
    
    p_h_left = r_h // 2
    p_h_right = r_h - p_h_left
    
    paddings = [[0, 0], [p_h_left, p_h_right], [p_w_left, p_w_right], [0, 0]]
    
    return conv_valid(name, tf.pad(x, paddings, name=name + "_pad"), kernel_shape, stride, channel, initializer, af)

def deconv_valid(name, x, kernel_shape, stride, output_shape, channel, initializer, af):
    '''
    add deconv2d layer / up-sampling layer with VALID padding method.

    note that channel is [input_channel, output_channel]

    output_size = (input_size - 1) * stride + kernel_size.
    '''

    #note that since conv2d_transpose does not accept
    #batch size of -1, the batch_size should be either
    
    #1.  tf.shape(x)[0]
    
    #or
    
    #2.  tf.stride_slice(tf.shape(x), [0], [1]), where x is the placeholder;
    
    #where x is the placeholder of input

    batch_size = tf.shape(x)[0]
        
    w = tf.get_variable(name + '_w', shape=(kernel_shape[0], kernel_shape[1], channel[1], channel[0]), initializer=initializer)
    b = tf.get_variable(name + '_b', shape=(channel[1],), initializer=tf.constant_initializer(0.0))
    
    return af(tf.nn.conv2d_transpose(x, w, output_shape=(batch_size, output_shape[0], output_shape[1], channel[1]), strides=(1, stride[0], stride[1], 1), padding='VALID') + b, name=name), w, b


def linear(name, x, channel, stddev=0.02, bias=0.0, af=None):
    '''
    matrix multiplication as full connect layer
    '''
    
    w = tf.get_variable(name + "_w", shape=[channel[0], channel[1]], initializer=tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable(name + "_b", shape=(channel[1]), initializer=tf.constant_initializer(bias))
    if af is None:
        return tf.add(tf.matmul(x, w), b, name=name), w, b
    else:
        return af(tf.add(tf.matmul(x, w), b, name=name + "_add"), name=name), w, b