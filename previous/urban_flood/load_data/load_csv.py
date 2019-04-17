import numpy as np
import pandas as pd
import cv2

'''
load csv data for luzern and hoengg case
'''

def load_expand_terrain_scale(path_terrain, path_label, patch_num, size_terrain, size_label, size_to, lowest=0, highest=2000, dtype_img=np.float32, dtype_label=np.uint8, seed=1):
    '''
    load raw csv data (full-scale image and label) and return the randomly sampled patches, patches that include invalid area (class = 0) are excluded

    parameters
    ------------------
    path_terrain: filename of terrain csv
    path_label: filename of label csv
    patch_num: number of patches to generate
    size_terrain: size of the terrain area, it should be larger than the size of the label area so that the machine may learn more from local patch
    size_label: size of the label area
    size_to: resize the terrain and the label patches to this value
    lowest: lowest level (meter) used to normalize the height between 0 and 1
    highest: highest level (meter) used to normalize the height between 0 and 1
    dtype_img: dtype of the terrain patch, default is float32, select float16 for less memory usage or float64 for higher percision
    dtype_label: dtype of the label patch, default is uint8
    seed: random seed for patch generation
    ------------------
    return: patches, labels, coords of the patch (left top corner of the max(size_terrain, size_label)), shape of the original image
    '''
    img_geo, img_gradient, img_label, mask = load_image(path_terrain, path_label, lowest, highest)
    img_all = np.transpose([img_geo,img_gradient[0],img_gradient[1]],[1,2,0])
    print(img_all.shape)

    # sample indices from data set (grid sample)
    patch_img = []
    patch_label = []
    patch_coord = []
    n = 0

    print("generating indices")
    np.random.seed(seed)

    patch_size = max(size_terrain,size_label)
    offset_terrain = (patch_size - size_terrain) // 2
    offset_label = (patch_size - size_label) // 2

    print("patch size:", patch_size)
    print("offset: t", offset_terrain, "l", offset_label)

    while n < patch_num:
        h = np.random.randint(0,img_geo.shape[0] - patch_size)
        w = np.random.randint(0,img_geo.shape[1] - patch_size)

        if not np.any(mask[h:h + patch_size,w:w + patch_size]):
            tmp = img_all[h + offset_terrain:h + offset_terrain + size_terrain,w + offset_terrain:w + offset_terrain + size_terrain]
            patch_img.append(cv2.resize(tmp,(size_to, size_to)).astype(dtype_img))
            tmp = img_label[h + offset_label :h + offset_label + size_label,w + offset_label:w + offset_label + size_label]
            patch_label.append(cv2.resize(tmp,(size_to, size_to)).astype(dtype_label))
            patch_coord.append([h, w])
            n +=1

    return np.array(patch_img),np.array(patch_label), np.array(patch_coord), img_geo.shape

def load_expand_terrain(path_terrain, path_label, patch_num, size_terrain, size_label, lowest=0, highest=2000, dtype_img=np.float32, dtype_label=np.uint8, seed=1):
    '''
    load raw csv data (full-scale image and label) and return the randomly sampled patches, patches that include invalid area (class = 0) are excluded

    parameters
    ------------------
    path_terrain: filename of terrain csv
    path_label: filename of label csv
    patch_num: number of patches to generate
    size_terrain: size of the terrain area, it should be larger than the size of the label area so that the machine may learn more from local patch
    size_label: size of the label area
    lowest: lowest level (meter) used to normalize the height between 0 and 1
    highest: highest level (meter) used to normalize the height between 0 and 1
    dtype_img: dtype of the terrain patch, default is float32, select float16 for less memory usage or float64 for higher percision
    dtype_label: dtype of the label patch, default is uint8
    seed: random seed for patch generation
    ------------------
    return: patches, labels, coords of the patch (top left corner), shape of the original image
    '''

    img_geo, img_gradient, img_label, mask = load_image(path_terrain, path_label, lowest, highest)
    img_all = np.transpose([img_geo,img_gradient[0],img_gradient[1]],[1,2,0])
    print(img_all.shape)

    # sample indices from data set (grid sample)
    patch_img = []
    patch_label = []
    patch_coord = []
    n = 0

    print("generating indices")
    np.random.seed(seed)

    patch_size = max(size_terrain,size_label)
    offset_terrain = (patch_size - size_terrain) // 2
    offset_label = (patch_size - size_label) // 2

    while n < patch_num:
        h = np.random.randint(0,img_geo.shape[0] - patch_size)
        w = np.random.randint(0,img_geo.shape[1] - patch_size)

        if not np.any(mask[h:h + patch_size,w:w + patch_size]):
            patch_img.append(img_all[h + offset_terrain:h + offset_terrain + size_terrain,w + offset_terrain:w + offset_terrain + size_terrain].astype(dtype_img))
            patch_label.append(img_label[h + offset_label :h + offset_label + size_label,w + offset_label:w + offset_label + size_label].astype(dtype_label))
            patch_coord.append([h, w])
            n +=1

    return np.array(patch_img),np.array(patch_label), np.array(patch_coord), img_geo.shape

def load_same_size(path_terrain, path_label, patch_num, patch_size, lowest=0, highest=2000, dtype_img=np.float32, dtype_label=np.uint8, seed=1):
    '''
    load raw csv data (full-scale image and label) and return the randomly sampled patches, patches that include invalid area (class = 0) are excluded

    parameters
    ------------------
    path_terrain: filename of terrain csv
    path_label: filename of label csv
    patch_num: number of patches to generate
    patch_size: size of patches
    lowest: lowest level (meter) used to normalize the height between 0 and 1
    highest: highest level (meter) used to normalize the height between 0 and 1
    dtype_img: dtype of the terrain patch, default is float32, select float16 for less memory usage or float64 for higher percision
    dtype_label: dtype of the label patch, default is uint8
    seed: random seed for patch generation
    ------------------
    return: patches, labels, coords of the patch (top left corner), shape of the original image
    '''
    return load_expand_terrain(path_terrain, path_label, patch_num, patch_size, patch_size, lowest, highest, dtype_img, dtype_label, seed)

def load_image(path_terrain, path_label,lowest, highest):
    # load terrain data
    terrain = pd.read_csv(path_terrain,header=None,dtype=np.float64)
    label = pd.read_csv(path_label,header=None,dtype=np.int16)
    img_geo = terrain.values

    # class 0 is not what we need
    mask = (label.values == 0)

    # class 1 - 4 is what we need, minus 1
    # note that when visualization, it is better to plus 1 back
    img_label = label.values - 1

    # get gradient, clip between -1 and 1
    img_gradient = np.gradient(img_geo)
    img_gradient[0][img_gradient[0] >= 1] = 1
    img_gradient[1][img_gradient[1] >= 1] = 1
    img_gradient[0][img_gradient[0] <= -1] = -1
    img_gradient[1][img_gradient[1] <= -1] = -1

    # normalize if lowest and highest are given
    if lowest is not None and highest is not None:
        img_geo[mask] = np.max(img_geo)
        img_geo = ((img_geo - lowest) / (highest - lowest) - 0.5) * 2

    return img_geo, img_gradient, img_label, mask

def load_terrain_scale(path_terrain, path_label, patch_num, terrain_size, img_size, height_delta_max, dtype_img=np.float32, dtype_label=np.uint8, seed=1):
    '''
    load a terrain patch and the water level at the center point of the patch

    parameters
    ------------------
    path_terrain: filename of terrain csv
    path_label: filename of label csv
    patch_num: number of patches to generate
    terrain_size: size of terrain
    img_size: size of terrain patch after resize
    height_delta_max: height difference within this value will be normalized to [1, -1]
    dtype_img: dtype of the terrain patch, default is float32, select float16 for less memory usage or float64 for higher percision
    dtype_label: dtype of the label patch, default is uint8
    seed: random seed for patch generation
    ------------------
    return: patches, labels, coords of the patch (center), shape of the original image
    '''
    img_geo, img_gradient, img_label, mask = load_image(path_terrain, path_label, None, None)
    # sample indices from data set (grid sample)
    patch_img = []
    patch_label = []
    patch_coord = []
    n = 0

    print("generating indices")
    np.random.seed(seed)

    offset = terrain_size // 2

    while n < patch_num:
        h = np.random.randint(0,img_geo.shape[0] - terrain_size)
        w = np.random.randint(0,img_geo.shape[1] - terrain_size)

        if not np.any(mask[h:h + terrain_size,w:w + terrain_size]):
            img = img_geo[h:h + terrain_size,w:w + terrain_size]
            img_g0 = img_gradient[0][h:h + terrain_size,w:w + terrain_size]
            img_g1 = img_gradient[1][h:h + terrain_size,w:w + terrain_size]

            h += offset
            w += offset

            height_center = img_geo[h,w]

            img = (img - height_center) / height_delta_max
            img[img > 1] = 1
            img[img < -1] = -1

            img_all = np.transpose([img,img_g0,img_g1],[1,2,0])

            patch_img.append(cv2.resize(img_all,(img_size, img_size)).astype(dtype_img))
            patch_label.append(img_label[h,w].astype(dtype_label))
            patch_coord.append([h, w])

            n +=1

    return np.array(patch_img),np.array(patch_label), np.array(patch_coord), img_geo.shape