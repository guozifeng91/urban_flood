import tensorflow as tf

import nets.segnet_old as segnet_old
import nets.segnet as segnet
import nets.unet as unet

def unet_sym_k7_2(input_channel, output_channel,img_size=256):
    tf.reset_default_graph()

    encode_channel = [32,64,128,128]
    encode_kernel_size = [[7,7],[7,7],[7,7],[7,7]]
    pooling_size = [[2,2],[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu

    return NNModel("unet_sym",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'encode_channel':encode_channel,
            'output_channel':output_channel,
            'encode_kernel_size':encode_kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af}, "kernel7_2")

def segnet_old_256(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet_old model of 256 patch size
    '''
    tf.reset_default_graph()
    
    net_channel = [32,64,128]
    kernel_size = [[3,3],[3,3],[3,3]]
    pooling_size = [[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet_old",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af})

def segnet_256(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet model of 256 patch size
    '''
    tf.reset_default_graph()

    net_channel = [32,64,128]
    kernel_size = [[3,3],[3,3],[3,3]]
    pooling_size = [[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af})

def segnet_256_k5(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet model of 256 patch size
    '''
    tf.reset_default_graph()

    net_channel = [32,64,128]
    kernel_size = [[5,5],[5,5],[5,5]]
    pooling_size = [[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af}, "kernel5")

def segnet_256_k5_2(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet model of 256 patch size
    '''
    tf.reset_default_graph()

    net_channel = [32,64,128,128]
    kernel_size = [[5,5],[5,5],[5,5],[5,5]]
    pooling_size = [[2,2],[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af}, "kernel5_2")
            
def segnet_256_k7_2(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet model of 256 patch size
    '''
    tf.reset_default_graph()

    net_channel = [32,64,128,128]
    kernel_size = [[7,7],[7,7],[7,7],[7,7]]
    pooling_size = [[2,2],[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af}, "kernel7_2")

def segnet_256_k37_2(input_channel, output_channel,img_size=256):
    '''
    this function reset the tensorflow graph and returns a default segnet model of 256 patch size
    '''
    tf.reset_default_graph()

    net_channel = [32,64,128,128]
    kernel_size = [[3,3],[7,7],[7,7],[7,7]]
    pooling_size = [[2,2],[2,2],[2,2],[2,2]]
    initializer = tf.contrib.layers.xavier_initializer()
    af = tf.nn.leaky_relu
    
    return NNModel("segnet",{'img_size':[img_size,img_size],
            'input_channel':input_channel,
            'net_channel':net_channel,
            'output_channel':output_channel,
            'kernel_size':kernel_size,
            'pooling_size':pooling_size,
            'initializer':initializer,
            'af':af}, "kernel37_2")
            
class NNModel:
    def __init__ (self, model_name, param, given_name="default"):
        self.name = model_name + "_" + given_name
        
        if model_name == "segnet_old":
            self.phs, self.ls, self.vs = segnet_old.prediction_network_new(param['img_size'],
                param['input_channel'],
                param['net_channel'],
                param['output_channel'],
                param['kernel_size'],
                param['pooling_size'],
                param['initializer'],
                param['af'])
        elif model_name == "segnet":
            self.phs, self.ls, self.vs = segnet.prediction_network(param['img_size'],
                param['input_channel'],
                param['net_channel'],
                param['output_channel'],
                param['kernel_size'],
                param['pooling_size'],
                param['initializer'],
                param['af'])
        elif model_name == 'unet_sym':
            self.phs, self.ls, self.vs = unet.unet_sym(param['img_size'],
                param['input_channel'],
                param['encode_channel'],
                param['output_channel'],
                param['encode_kernel_size'],
                param['pooling_size'],
                param['initializer'],
                param['af'])
        elif model_name == 'unet':
            self.phs, self.ls, self.vs = unet.unet(param['img_size'],
                param['input_channel'],
                param['encode_channel'],
                param['latent_channel'],
                param['decode_channel'],
                param['output_channel'],
                param['encode_kernel_size'],
                param['latent_kernel_size'],
                param['decode_kernel_size'],
                param['pooling_size'],
                param['initializer'],
                param['af'])
        else:
            raise Exception("Cannot recognize", name)
            
    def get_model():
        return self.phs, self.ls, self.vs
        