import numpy as np
import gzip
import pandas as pd
import os
from os.path import join

import featureExtraction as fe
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

channels_rescale_dem_zero_one = 2
def rescale_dem_zero_one(dem_array):
    # rescale dem_array
    mask_indice = dem_array > 0
    
    vmin=dem_array[mask_indice].min()
    vmax=dem_array[mask_indice].max()
    
    dem_array[mask_indice] -= vmin
    dem_array[mask_indice] /= (vmax - vmin)
    
    # fill non-data areas
    mask_indice = dem_array < 0
    dem_array[mask_indice] = 0
    
    # mask
    mask_array = np.ones_like(dem_array,dtype=np.float32)
    mask_array[mask_indice] = -1
    
    # final output of dem
    return np.transpose(np.array([dem_array,mask_array]),[1,2,0])

channels_rescale_dem_negative_positive = 2
def rescale_dem_negative_positive(dem_array):
    # rescale dem_array
    mask_indice = dem_array < 0
    
    dem_array[dem_array>0] -= dem_array[dem_array>0].mean()
    dem_array[mask_indice] = 0
    dem_array *= 0.1
    
    # mask
    mask_array = np.ones_like(dem_array,dtype=np.float32)
    mask_array[mask_indice] = -1
    
    # final output of dem
    return np.transpose(np.array([dem_array,mask_array]),[1,2,0])
    
channels_generate_features = 6
def generate_features(dem_array):
    '''
    generate the necessary features from the dem
    
    the result will be an nd array [height, width, channel]
    '''
        
    print("generating features")
    
    slop,curvature,cos,sin,_ = fe.feature_lin_mean(dem_array,1)
    
    # rescale dem_array
    mask_indice = dem_array > 0
    
    vmin=dem_array[mask_indice].min()
    vmax=dem_array[mask_indice].max()
    
    dem_array[mask_indice] -= vmin
    dem_array[mask_indice] /= (vmax - vmin)
    
    # fill non-data areas
    mask_indice = dem_array < 0
    dem_array[mask_indice] = 0
    
    # mask
    mask_array = np.ones_like(dem_array,dtype=np.float32)
    mask_array[mask_indice] = -1
    
    # final output of dem
    return np.transpose(np.array([dem_array,mask_array,slop,cos,sin,curvature]),[1,2,0])