import tensorflow as tf
import numpy as np

class NpyParser_TF:
    '''
    parse npy format with tensorflow to include this format into the pipeline of tensorflow dataset.
    
    usage:
    
    1. new an parse object by given the expected shape and dtype of npy file
    (this is limited due to the 1.10 version of tensorflow, with higher version, a more powerful parser is possible)
    
    parser = NpyParse_TF((shape1,shape2,...), tf.float32)
    
    2. set the parser as the map of a dataset:
    
    dataset = tf.data.Dataset.from_tensor_slices(list of filenames)
    dataset = dataset.map(parser.parse)
    
    3. that's it, using the dataset with iterator gives the contents of the npy files 
    
    '''
    def __init__(self, shape, dtype):
        '''
        create a NpyParse_TF object by giving the expected shape and dtype
        '''
        self.shape = shape
        self.dtype = dtype
        self.size = 1
        for s in shape:
            self.size *= s
    
    def parse(self, npy_file):
        '''
        parse the given file using tensorflow operations 
        '''
        arr = tf.decode_raw(tf.read_file(npy_file),self.dtype)
        arr = arr[-self.size:]
        return tf.reshape(arr,self.shape)

class ImageParser_TF:
    def __init__(self,shape):
        self.shape = shape
        
    def parse(self, img_file):
        return tf.reshape(tf.decode_image(tf.read_file(img_file)),shape=self.shape)

def load_dataset_npy(filenames, shape, dtype, num_parallel_calls = 2):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(NpyParser_TF(shape,dtype).parse, num_parallel_calls = num_parallel_calls)
        
    return dataset