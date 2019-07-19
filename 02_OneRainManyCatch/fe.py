import numpy as np
import scipy.ndimage

# feature extractors of the input array
# the extracted features are concatenated in the last axis (producing a multi-channel image)

# list of classes:

# 1. Features. A wrapper for all the feature extractor, pass a list of extractors to build a final extractor
# 2. GeographicByDilate. extract geographic features, can pass keep_channels to choose which channel shall be kept
# 3. GeographicByKernel. another implementation of extract geographic features
# 4. NormalizedHeight. normalize DEM based on local and global mean value
# 5. Mask. identify the no-data area
# 6. OneHot. one-hot encoder
# 7. SoftOneHot. one-hot encoder in floating values


DEBUG = True

class Features:
    '''
    a set of feature extractors, initiated by a list of instances and returns the flattened features
    '''
    def __init__(self, list_of_fe):
        self.list_of_fe = list_of_fe
    
    def name(self):
        name = ""
        for fe in self.list_of_fe:
            name += fe.name()
        return name
    
    def channels(self):
        # get the total channel of all feature extractors
        return sum([fe.channels() for fe in self.list_of_fe])
    
    def features(self, dem_array):
        # run all feature extractor and flatten the result
        arr = [fe.features(dem_array) for fe in self.list_of_fe]
        
        if DEBUG:
            for a in arr:
                print (a.shape, a.dtype)
            
        # return as a list so that the input can be concatenated with the output channels
        return np.concatenate(arr,axis=-1)

def feature_fast_mean(array_all, group_size, cell_size=1):
    '''
    calculate the slop, curvature and aspect of a terrain.
    faster implementation, use n-dimensional array calculation instead of pixel-wised calculation.
    -----
    
    about group_size:
    
    the DEM features of one pixel are calculated based on its 8 neighbors
    
    assume we merge multiple pixels into one bigger pixel, and do the same calculation,
    we get the result with the same meaning but in larger scale.
    
    the group_size indicates how many pixels are merged into one big pixel.
    or, the size of the pixel group = group_size x group_size.
    
    the mean value of each pixel group is obtained by scipy.ndimage.filters.uniform_filter
    
    the no-data pixels are filled by dilation
    
    to locate the center pixel faster, we define that the group_size must be an odd number.
    '''
    array_pad=np.pad(array_all,group_size,'edge')
    
    # fill the no-data area by dilation
    dilate_size = group_size * 3
    dilate_array = scipy.ndimage.grey_dilation(array_pad,size=(dilate_size,dilate_size))
    
    # override have-data area with the original array
    # the effect looks like offet the original array around its boundary
    indice=array_pad>0
    dilate_array[indice]=array_pad[indice]
    
    # apply mean filter
    dilate_array = scipy.ndimage.filters.uniform_filter(dilate_array,size=(group_size,group_size))
    
    group_size2=group_size+group_size
    height, width = array_all.shape

    # first row
    a=dilate_array[0:height,0:width]
    b=dilate_array[0:height,group_size:group_size+width]
    c=dilate_array[0:height,group_size2:group_size2+width]
    
    # second row
    d=dilate_array[group_size:group_size+height,0:width]
    e=dilate_array[group_size:group_size+height,group_size:group_size+width]
    f=dilate_array[group_size:group_size+height,group_size2:group_size2+width]
    
    # third row
    g=dilate_array[group_size2:group_size2+height,0:width]
    h=dilate_array[group_size2:group_size2+height,group_size:group_size+width]
    i=dilate_array[group_size2:group_size2+height,group_size2:group_size2+width]
    # as the array was dilated, no no-data should exist for pixels correspond to the pixels of e>0
    
    del(indice)
    del(array_pad)
    
    # the actual size of the pixel group (pixel num * pixel size)
    group_size = cell_size * group_size
    
    dx= ((c + 2*f + i) - (a + 2*d + g)) / (8*group_size)
    dy= ((g + 2*h + i) - (a + 2*b + c)) / (8*group_size)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    slop = np.arctan(np.sqrt(dx**2 + dy**2)) # in radian
    
    # desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
    aspect = np.arctan2(dy, -dx)
    
    del(dx)
    del(dy)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    group_size_sq = group_size**2
    
    D = ((d + f) /2 - e) / (group_size_sq)
    E = ((b + h) /2 - e) / (group_size_sq)
    # curvature = -200 * (D + E) # why 200 here?
    
    curvature = np.clip(-group_size * (D + E),-4,4) # rescale and clip the curvature using group_size (just a trick)
    
    del(D)
    del(E)
    
    # set features of invalid pixels to 0
    indice = array_all<0
    
    cos = np.cos(aspect)
    sin = np.sin(aspect)
    
    slop[indice]=0
    cos[indice]=0
    sin[indice]=0
    curvature[indice]=0
    aspect[indice]=0
    return slop,curvature,cos,sin,aspect

def feature_exp_mean(array_all,level,cell_size=1):
    '''
    the size of the group_size growths exponentially regarding to level
    '''
    return feature_fast_mean(array_all, 3**(level-1),cell_size)

def feature_lin_mean(array_all,level,cell_size=1):
    '''
    the size of the group_size growths linearly regarding to level
    '''
    return feature_fast_mean(array_all, level*2-1,cell_size)
    
class GeographicByDilate:
    def __init__(self,level, method="l", cell_size=1, keep_channels = [0,1,2,3,4]):
        '''
        method: l for lin or e for exp
        channels: slop,curvature,cos,sin,aspect
        '''
        self.level = level
        self.method = method
        self.cell_size = cell_size
        self.keep_channels = keep_channels
    
    def name(self):
        return "GD" + str(self.level) + self.method + str(len(self.keep_channels))
    
    def channels(self):
        return len(self.keep_channels)
    
    def features(self, dem_array):
        result = feature_lin_mean(dem_array, self.level, self.cell_size) if self.method == "l" else feature_exp_mean(dem_array, self.level, self.cell_size)
        result = [result[c] for c in self.keep_channels]
        return np.transpose(result,[1,2,0])
        
def get_features_conv(array_all,kernel_size,kernel_dist,cell_size=1):
    if kernel_size % 2 == 0:
        kernel_size += 1
        print ("kernel size increased to",kernel_size)
    # pad
    pad_size = (kernel_size-1)//2 + kernel_dist
    pad_size2 = pad_size + pad_size
    
    height, width = array_all.shape
    
    array_pad=np.pad(array_all,pad_size,'edge')
    # another idea regarding the no-data areas:
    # fill with min elevation since the water goes out of the domain when running simulations
    
    # fill with nan and use convolve to calculate the uniform filter
    array_pad[array_pad<0] = np.nan
    if kernel_size>1:
        array_pad = scipy.ndimage.convolve(array_pad, np.ones((kernel_size,kernel_size))/(kernel_size**2))
    
    # 9 pixels, represented as a to i
    
    # first row
    a=np.copy(array_pad[0:height,0:width])
    b=np.copy(array_pad[0:height,pad_size:pad_size+width])
    c=np.copy(array_pad[0:height,pad_size2:pad_size2+width])
    
    # second row
    d=np.copy(array_pad[pad_size:pad_size+height,0:width])
    e=array_pad[pad_size:pad_size+height,pad_size:pad_size+width]
    f=np.copy(array_pad[pad_size:pad_size+height,pad_size2:pad_size2+width])
    
    # third row
    g=np.copy(array_pad[pad_size2:pad_size2+height,0:width])
    h=np.copy(array_pad[pad_size2:pad_size2+height,pad_size:pad_size+width])
    i=np.copy(array_pad[pad_size2:pad_size2+height,pad_size2:pad_size2+width])
    
    # replace the center nan (e) as the original value from array_all
    indice = np.isnan(e)
    e[indice] = array_all[indice]
    
    # replace other nans (a,b,c ...) as the center value (e)
    for pixel in [a,b,c,d,f,g,h,i]:
        indice = np.isnan(pixel)
        pixel[indice]=e[indice]
    
    # calculate features
    group_size = cell_size * kernel_dist
    dx= ((c + 2*f + i) - (a + 2*d + g)) / (8*group_size)
    dy= ((g + 2*h + i) - (a + 2*b + c)) / (8*group_size)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) # in radian
    # desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
    aspect = np.arctan2(dy, -dx)
    
    del(dx)
    del(dy)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    group_size_sq = group_size**2
    
    D = ((d + f) /2 - e) / (group_size_sq)
    E = ((b + h) /2 - e) / (group_size_sq)
    # curvature = -200 * (D + E) # why 200 here?
    
    curvature = np.clip(-2 *(D + E),-4,4)
    
    del(D)
    del(E)
    
    # delete invalid data
    indice = array_all<0
    
    slope[indice] = 0
    aspect[indice] = 0
    curvature[indice] = 0
    return slope,aspect,curvature

def get_features_filter(array_all,kernel_size,kernel_dist,cell_size=1):
    if kernel_size % 2 == 0:
        kernel_size += 1
        print ("kernel size increased to",kernel_size)
    # pad
    pad_size = (kernel_size-1)//2 + kernel_dist
    pad_size2 = pad_size + pad_size
    
    height, width = array_all.shape
    
    array_pad=np.pad(array_all,pad_size,'edge')
    # another idea regarding the no-data areas:
    # fill with min elevation since the water goes out of the domain when running simulations
    
    # fill with nan and use convolve to calculate the uniform filter
    array_pad[array_pad<0] = np.min(array_all[array_all>0])
    
    if kernel_size > 1:
        array_pad = scipy.ndimage.filters.uniform_filter(array_pad,size=(kernel_size,kernel_size))
    
    # 9 pixels, represented as a to i
    
    # first row
    a=array_pad[0:height,0:width]
    b=array_pad[0:height,pad_size:pad_size+width]
    c=array_pad[0:height,pad_size2:pad_size2+width]
    
    # second row
    d=array_pad[pad_size:pad_size+height,0:width]
    e=array_pad[pad_size:pad_size+height,pad_size:pad_size+width]
    f=array_pad[pad_size:pad_size+height,pad_size2:pad_size2+width]
    
    # third row
    g=array_pad[pad_size2:pad_size2+height,0:width]
    h=array_pad[pad_size2:pad_size2+height,pad_size:pad_size+width]
    i=array_pad[pad_size2:pad_size2+height,pad_size2:pad_size2+width]
    
    # calculate features
    group_size = cell_size * kernel_dist
    dx= ((c + 2*f + i) - (a + 2*d + g)) / (8*group_size)
    dy= ((g + 2*h + i) - (a + 2*b + c)) / (8*group_size)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) # in radian
    # desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
    aspect = np.arctan2(dy, -dx)
    
    del(dx)
    del(dy)
    
    # http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-curvature-works.htm
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    group_size_sq = group_size**2
    
    D = ((d + f) /2 - e) / (group_size_sq)
    E = ((b + h) /2 - e) / (group_size_sq)
    # curvature = -200 * (D + E) # why 200 here?
    
    curvature = np.clip(-2 *(D + E),-4,4)
    
    del(D)
    del(E)
    
    # delete invalid data
    indice = array_all<0
    
    slope[indice] = 0
    aspect[indice] = 0
    curvature[indice] = 0
    return slope,aspect,curvature

class GeographicByKernel:
    def __init__(self, kernel_size, kernel_dist, method="f", cell_size=1):
        '''
        method: f for filter or c for conv
        '''
        self.kernel_size = kernel_size
        self.kernel_dist = kernel_dist
        self.method = method
        self.cell_size = cell_size
    
    def name(self):
        return "GK" + str(self.kernel_size) + "-" + str(self.kernel_dist) + self.method
    
    def channels(self):
        return 3
    
    def features(self, dem_array):
        if self.method == "f":
            return np.transpose(get_features_filter(dem_array, self.kernel_size, self.kernel_dist, self.cell_size),[1,2,0])
        else:
            return np.transpose(get_features_conv(dem_array, self.kernel_size, self.kernel_dist, self.cell_size),[1,2,0])
            
def feature_dem_normalize(dem_array, size_list):
    # make a copy so the original data are not effected
    dem_array = np.copy(dem_array)
    
    # has-data and no-data areas
    indice = dem_array < 0
    indice_inv = dem_array > 0
    
    # fill no-data areas with the min value
    dem_array[indice] = np.min(dem_array[indice_inv>0])

    # local normalization
    arr = [dem_array - scipy.ndimage.filters.uniform_filter(dem_array,size=(k,k)) for k in size_list]
    
    # global normalization
    dem_array -= dem_array[indice_inv].mean()
    
    # fill the no-data area with 0
    arr.insert(0, dem_array)
    for a in arr:
        a[indice]=0
    
    return arr

class NormalizedHeight:
    '''
    translate the height by the mean height of the catchment, then rescale
    '''
    def __init__(self, size_list, scale=0.01):
        '''
        pass an empty list to self_list if only global normalization is needed
        '''
        self.size_list = size_list
        self.scale = scale
        
    def name(self):
        return "NH" + str(len(self.size_list))

    def channels(self):
        return 1 + len(self.size_list)
        
    def features(self, dem_array):
        return self.scale * np.transpose(feature_dem_normalize(dem_array, self.size_list),[1,2,0])

class RescaledHeight:
    '''
    rescale the height
    '''
    def __init__(self, scale=0.01, bias = 0):
        '''
        pass an empty list to self_list if only global normalization is needed
        '''
        self.scale = scale
        self.bias = bias
        
    def name(self):
        return "RS"+str(self.scale) + "-" + str(self.bias)

    def channels(self):
        return 1
        
    def features(self, dem_array):
        indice = dem_array < 0
        dem_array = self.bias + self.scale * dem_array
        dem_array[indice]=0
        return dem_array[...,np.newaxis]

class HeightToDepth:
    '''
    translate the height by maximum height of the catchment (max - height),
    convert the height map to depth map (as the water accumulates from the top to the bottom)
    '''
    def __init__(self, scale=0.01):
        '''
        pass an empty list to self_list if only global normalization is needed
        '''
        self.scale = scale
        
    def name(self):
        return "HD"+str(self.scale)

    def channels(self):
        return 1
        
    def features(self, dem_array):
        indice = dem_array < 0
        # depth max (all positive, 0 is the highest point, which has the lowest depth)
        dem_array = self.scale * (dem_array.max() - dem_array)
        # fill no-data as 0
        dem_array[indice]=0
        
        return dem_array[...,np.newaxis]
        
class Mask:
    def __init__(self):
        pass
    
    def name(self):
        return "M"
    
    def channels(self):
        return 1
    
    def features(self, dem_array):
        '''
        original dem must be given
        '''
        # mask
        mask_array = np.ones_like(dem_array,dtype=np.float32)
        mask_array[dem_array<0] = -1
        return mask_array[...,np.newaxis]

class OneHot:
    '''
    output feature extractor, one-hot
    
    this class has been tested
    '''
    def __init__(self, min_val, max_val, band_num):
        self.min_val = min_val
        self.max_val = max_val
        self.band_num = band_num
        self.step = (max_val - min_val) / band_num
    
    def name(self):
        return "OH"
    
    def channels(self):
        return self.band_num
    
    def features(self, dem_array):
        dem_int = np.floor((np.clip(dem_array, self.min_val, self.max_val-self.step*0.001) - self.min_val) / self.step)
        # return as a list of single-channel image
        # return [(dem_int==i).astype(np.uint32) for i in range(self.band_num)]
        
        # transposed result, see https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
        # return as one multi-channel image
        return (np.arange(dem_int.max()+1) == dem_int[...,None]).astype(np.uint8)
        
class SoftOneHot:
    '''
    soft one-hot, use one hot to represent the large-range continuous value
    
    for example, if the abs value is 2.7
    
    then the class "2" get activation of 0.3 and class "3" get activation of 0.7
    
    this class has been tested
    '''
    def __init__(self, min_val, max_val, band_num):
        self.min_val = min_val
        self.max_val = max_val
        self.band_num = band_num
        self.step = (max_val - min_val) / band_num
    
    def name(self):
        return "SOH"
    
    def channels(self):
        return self.band_num+1
    
    def features(self, dem_array):
        '''
        original dem must be given
        '''
        dem_float = (np.clip(dem_array, self.min_val, self.max_val) - self.min_val) / self.step
        dem_int = np.floor(dem_float) # integer part (current class)
        dem_float = dem_float - dem_int # decimal part (activation of next class)
        dem_float_inv = 1-dem_float # decimal part (activation of current class)

        # union(decimal of previous class, decimal of current class)
        # note that the operation "+" does the union
        return np.transpose([(dem_int==i).astype(np.uint32)*dem_float_inv + (dem_int==(i-1)).astype(np.uint32)*dem_float for i in range(self.band_num+1)],[1,2,0])