import tensorflow as tf
import numpy as np

class NpyParse_TF:
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
    def __init__(self, shape, dtype, fortran_order=False, little_endian=True):
        '''
        create a NpyParse_TF object by giving the expected shape and dtype
        
        be careful with the fortran_order and little_endian options
        
        fortran_order: the bytes of the nd-array are stored in a reversed order,
                       e.g., shape(a,b,c) cannot be reshaped using reshape(a,b,c),
                       but have to use transpose(reshape(c,b,a))
        
        little_endian (will be included soon): see https://en.wikipedia.org/wiki/Endianness
        
        currently these options cannot be detected automatically due to insufficient
        string functions provided by tensorflow, therefore please carefully check the
        header of input npy files from which you can get correct options.
        '''
        self.little_endian = little_endian
        self.dtype = dtype
        self.fortran_order=fortran_order
        self.size = 1
        for s in shape:
            self.size *= s
            
        if self.fortran_order:
            self.shape = list(reversed(shape))
        else:
            self.shape = shape
    
    def parse(self, npy_file):
        '''
        parse the given file using tensorflow operations 
        '''
        arr = tf.decode_raw(tf.read_file(npy_file),self.dtype,little_endian=self.little_endian)
        arr = arr[-self.size:]
        
        if self.fortran_order:
            return tf.transpose(tf.reshape(arr,self.shape))
        else:
            return tf.reshape(arr,self.shape)

class ImageParse_TF:
    def __init__(self,shape):
        self.shape = shape
        
    def parse(self, img_file):
        return tf.reshape(tf.decode_image(tf.read_file(img_file)),shape=self.shape)

def load_dataset_npy(filenames, shape, dtype, fortran_order=False, little_endian=True, num_parallel_calls = 2):
    '''
    create a tensorflow dataset object that read and parse npy files
    
    be careful with the fortran_order and little_endian options
    
    fortran_order: the bytes of the nd-array are stored in a reversed order,
                   e.g., shape(a,b,c) cannot be reshaped using reshape(a,b,c),
                   but have to use transpose(reshape(c,b,a))
    
    little_endian (will be included soon): see https://en.wikipedia.org/wiki/Endianness
    '''
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(NpyParse_TF(shape,dtype,fortran_order).parse, num_parallel_calls = num_parallel_calls)
    return dataset
    
import gzip
import pandas as pd
import os
from os.path import join

# asc loading functions
def read_ASC(file, skip_last_col=False):
    if skip_last_col:
        return pd.read_csv(file,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values[:,:-1]
    else:
        return pd.read_csv(file,header=None,skiprows=6,delimiter=" ",skipinitialspace =True,dtype=np.float32).values
    
def read_CSV(file, delimiter=",", skip_last_col=False):
    if skip_last_col:
        return pd.read_csv(file,header=None,skiprows=0,delimiter=delimiter,skipinitialspace =True).values[:,:-1]
    else:
        return pd.read_csv(file,header=None,skiprows=0,delimiter=delimiter,skipinitialspace =True).values

def rainfall_pattern(rain_file):
    data = pd.read_csv(rain_file,header=None,index_col=0,delimiter="\t",skipinitialspace =True)
    values = data.values
    return {i:data.index[i] for i in range(len(data.index))},values

def load_waterdepth(foldername, pattern_name_list):
    '''
    load the simulation results from a specific folder
    
    the result will be a list of numpy array
    '''
    files = os.listdir(foldername)
    waterdepth = []
    
    for pattern_name in pattern_name_list:
        pattern_file = None
        for f in files:
            if "_" + pattern_name + "_" in f:
                pattern_file = f
                break
        if pattern_file is None:
            raise Exception("cannot find simulation result for pattern " + pattern_name)
            
        print("loading",pattern_file)
        
        if ".gz" in pattern_file:
            with gzip.open(join(foldername,pattern_file),"rb") as pattern_file_gzip:
                wd = read_ASC(pattern_file_gzip)
                # some asc file contains empty columns, delete them
                if np.any(np.isnan(wd[:,-1])):
                    wd = wd[:,:-1]
                print (wd.shape)
                # we define that the prediction value for invalid area is always 0
                wd[wd<0]=0
                waterdepth.append(wd)
        else:
            wd = read_ASC(join(foldername,pattern_file))
            # some asc file contains empty columns, delete them
            if np.any(np.isnan(wd[:,-1])):
                wd = wd[:,:-1]
            print (wd.shape)
            # we define that the prediction value for invalid area is always 0
            wd[wd<0]=0
            waterdepth.append(wd)
    
    if len(waterdepth) == 1:
        return waterdepth[0]
    else:
        return np.transpose(np.array(waterdepth),[1,2,0])

def load_dem(dem_file):
    print("loading", dem_file)
    dem_array = None
    if ".gz" in dem_file:
        with gzip.open(dem_file,"rb") as zipfile:
            dem_array = read_ASC(zipfile,skip_last_col=False)
    else:
        dem_array = read_ASC(dem_file,skip_last_col=False)
    
    # some asc file contains empty columns, delete them
    if np.any(np.isnan(dem_array[:,-1])):
        dem_array = dem_array[:,:-1]
    print (dem_array.shape)
    return dem_array

def get_folders_from_root(root_folder, seed=1, test_set_propotion=3):
    # ===================read patches list===================
    patch_folders = np.array([d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder,d))])
    print(patch_folders)
    
    # shuffle the catchment folders
    np.random.seed(seed)
    np.random.shuffle(patch_folders)
    num_folders = patch_folders.shape[0]
    
    # use the first 1/base_num catchments as test data
    train_folders = patch_folders[max(1,num_folders//test_set_propotion):]
    test_folders = patch_folders[:max(1,num_folders//test_set_propotion)]
    print("test_folders", test_folders)
    return train_folders, test_folders

def get_folders_from_txt(txt, seed=1, test_set_propotion=3):
    # ===================read patches list===================
    with open(txt, "r") as text_file:
        patch_folders = text_file.read().split('\n')
    # remove empty lines
    patch_folders = list(filter(None, patch_folders))
    print(patch_folders)
    
    # shuffle the catchment folders
    np.random.seed(seed)
    np.random.shuffle(patch_folders)
    num_folders = patch_folders.shape[0]
    
    # use the first 1/base_num catchments as test data
    train_folders = patch_folders[max(1,num_folders//test_set_propotion):]
    test_folders = patch_folders[:max(1,num_folders//test_set_propotion)]
    print("test_folders", test_folders)
    return train_folders, test_folders

class PatchFromFile:
    '''
    access patch from pre-processed npy files
    
    be careful with the fortran_order and little_endian options
    
    fortran_order: the bytes of the nd-array are stored in a reversed order,
                   e.g., shape(a,b,c) cannot be reshaped using reshape(a,b,c),
                   but have to use transpose(reshape(c,b,a))
    
    little_endian (will be included soon): see https://en.wikipedia.org/wiki/Endianness
    '''
    def __init__ (self, files, shape, dtype, fortran_order=False, batch_size=8, repeat = None, parallel=2, shuffle=None, prefetch=1):
        data_loader = load_dataset_npy(files,shape,dtype,fortran_order=fortran_order, num_parallel_calls=parallel)
        if shuffle is not None:
            data_loader = data_loader.shuffle(shuffle)
        
        self.patch_num = len(files)
        self.current = 0 # counter for end_of_epoch(), an optional function that does not effect the patch generation
        self.batch_size = batch_size
        
        data_loader = data_loader.batch(batch_size).prefetch(prefetch).repeat(repeat)
        self.data_loader = data_loader
        self.get_next_op = data_loader.make_one_shot_iterator().get_next()
    
    def set_session(self, sess):
        self.sess = sess
    
    def next_batch(self):
        arr = self.sess.run(self.get_next_op)
        self.current += len(arr)
        return arr
    
    def end_of_epoch(self):
        '''
        indicate if current epoch has completed, this is an indicator function to simplify the
        training code, it does not effect the patch generation
        
        call end_of_epoch before next_batch to get correct result
        
        for example:
        
        while not end_of_epoch():
            next_batch()
            ...
        '''
        if self.current >= self.patch_num:
            self.current = self.current % self.patch_num
            return True
        else:
            return False

def generate_patch_location(dem_array, patch_size, key_channel, key_value, patch_num, ratio):
    height, width, _ = dem_array.shape
    
    if patch_num is None:
        patch_num = ratio * (height * width) // (patch_size**2)

    patches = []
    i=0
    while i < patch_num:
        w = np.random.randint(width - patch_size)
        h = np.random.randint(height - patch_size)
        
        if np.all(dem_array[h:h+patch_size,w:w+patch_size,key_channel]==key_value):
            # not any 1 within the area?
            continue
        
        i+=1
        patches.append([h,w])
    return np.array(patches, dtype=np.uint32)
    
class PatchFromImage:
    '''
    generate patch from pre-loaded multi-channel images
    
    this patch generator repeats without limits
    
    note: the input image must be rank 3 image, even there is only one channel!
    '''
    def __init__(self, image_list, patch_size, key_channel, key_value, num = None, ratio = 5, batch_size=8, seed=1, shuffle=True, augmentation=False):
        # set seed
        if seed is not None:
            np.random.seed(seed)
            
        self.image_list = image_list
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.current = 0 # counter for end_of_epoch(), an optional function that does not effect the patch generation
        
        # patch location for each nd_image
        # locs is a 2 level list, the 1st level indicates which image, the 2nd level indicates which patch
        locs = [generate_patch_location(arr, patch_size, key_channel, key_value, num, ratio) for arr in image_list]
        
        # insert the index of the corresponding image to the end of the patch location
        # locs is a 2 level list, the 2nd level has length of 3
        locs = [np.concatenate([locs[i], i * np.ones((len(locs[i]),1),dtype=np.uint32)], axis=-1) for i in range(len(locs))]
        
        # flatten the list to one level only
        self.patch_locations = np.concatenate(locs,axis=0)
        
        # data augmentation, insert x and y flip direction at the end of the patch location
        self.augmentation=augmentation
        if augmentation:
            self.patch_locations=np.array([[h,w,i,y,x] for x in [-1,1] for y in [-1,1] for h,w,i in self.patch_locations])
            
        self.patch_num = len(self.patch_locations)
        
        print (self.patch_num, "patches in total")
        # initiate the indice
        self.do_shuffle()

    def do_shuffle(self):
        '''
        shuffle the patch locations in axis 0
        '''
        if self.shuffle:
            np.random.shuffle(self.patch_locations)
        self.start = 0
        
    def next_batch(self, debug=False):
        '''
        return the next batch of patches
        '''
        # if the indicator exceed the patch number, reshuffle the patches and reset the indicator
        if self.start>= self.patch_num:
            self.do_shuffle()
            if debug:
                print("end of list")
        
        # read next batch
        start = self.start # start of the indice (included)
        end = min(start+self.batch_size,self.patch_num) # end of the indice (excluded)
        # the batch may less than batch_size when reaching the end of the patch list 

        self.current += end-start
        
        # move the indicator to the next batch
        self.start += self.batch_size
        
        locs = self.patch_locations[start:end]
        
        if debug:
            return locs
        else:
            #sample from the image list and return the nd-array
            if self.augmentation:
                # take 4 types of flip into consideration
                return np.array([self.image_list[i][h:h+self.patch_size, w:w+self.patch_size][::y,::x] for h,w,i,y,x in locs])
            else:
                return np.array([self.image_list[i][h:h+self.patch_size, w:w+self.patch_size] for h,w,i in locs])
    
    def end_of_epoch(self):
        '''
        indicate if current epoch has completed, this is an indicator function to simplify the
        training code, it does not effect the patch generation
        
        call end_of_epoch before next_batch to get correct result
        
        for example:
        
        while not end_of_epoch():
            next_batch()
            ...
        '''
        if self.current >= self.patch_num:
            self.current = 0
            return True
        else:
            return False