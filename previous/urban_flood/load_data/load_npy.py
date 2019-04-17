import numpy as np

'''
load npy data for portugal case
'''

def load_terrain_grid(path_terrain, path_label, path_pattern, patch_size, dtype_img=np.float32, dtype_label=np.float32):
    '''
    @deprecated
    load the terrain data and split them into grid
    '''
    img_all = np.load(path_terrain)
    img_label = np.load(path_label)

    rain_pattern = np.loadtxt(path_pattern,delimiter='\t')

    channel,height,width = img_all.shape
    patch_img = []
    patch_label = []
    patch_coord = []

    h_num = height // patch_size
    w_num = width // patch_size

    for h in range(h_num):
        for w in range(w_num):
            patch_img.append(np.transpose(img_all[:,h*patch_size:(1+h)*patch_size,w*patch_size:(1+w)*patch_size],[1,2,0]).astype(dtype_img))
            patch_label.append(img_label[:,h*patch_size:(1+h)*patch_size,w*patch_size:(1+w)*patch_size].astype(dtype_label))
    
    return np.array(patch_img), np.array(patch_label), rain_pattern,[h_num,w_num], [height, width]


def load_terrain(path_terrain, path_label, path_pattern, patch_num, patch_size, dtype_img=np.float32, dtype_label=np.float32, seed=1):
    '''
    @deprecated
    load terrain data and make random sampling

    channels regarding to the terrain patch: [terrain, mask, slop, aspect cos, aspect sin, curvature]

    the sequence of rain pattern in path label:
    2-1, 5-1, 10-1, ..., 50-3, 100-3

    returned value:
    ----------------------
    patch_image, ndarray[patch_num, h, w, c]
    patch_label, ndarray[patch_num, pattern_num, h, w]
    rain_pattern, ndarray[pattern_num, array]
    patch_coord, ndarray[patch_num, coord]
    [height, width]
    '''
    img_all = np.load(path_terrain)
    img_label = np.load(path_label)
    rain_pattern = np.loadtxt(path_pattern,delimiter='\t')
    print(rain_pattern)

    channel,height,width = img_all.shape
    print(img_all.shape)

    # sample indices from data set (grid sample)
    patch_img = []
    patch_label = []
    patch_coord = []
    n = 0

    print("generating indices")
    np.random.seed(seed)

    while n < patch_num:
        h = np.random.randint(0,height - patch_size)
        w = np.random.randint(0,width - patch_size)

        if np.any(img_all[1, h:h + patch_size,w:w + patch_size]):
            patch_img.append(np.transpose(img_all[:,h:h + patch_size,w:w + patch_size],[1,2,0]).astype(dtype_img))
            patch_label.append(img_label[:,h:h + patch_size,w :w + patch_size].astype(dtype_label))
            patch_coord.append([h, w])
            n +=1

    return np.array(patch_img), np.array(patch_label), rain_pattern, np.array(patch_coord), [height, width]

def get_patch_indice_random(height, width, patch_num, patch_size, mask_terrain=None, seed=None):
    '''
    get random patch indices

    parameters:
    ------------------
    height
    width
    patch_num: number of patches
    patch_size: size of the patch
    mask_terrain: numpy array with shape [height, width], indicating the valid area of the terrain
    seed: random seed

    return:
    ------------------
    list with the shape of [patch_num, 4], the elements are [minH, maxH, minW, maxW]
    '''
    patch_indice = []
    n = 0

    # set random seed
    if seed is not None:
        np.random.seed(seed)

    while n < patch_num:
        h = np.random.randint(0,height - patch_size)
        w = np.random.randint(0,width - patch_size)

        if mask_terrain is None or np.any(mask_terrain[h:h + patch_size,w:w + patch_size]):
            patch_indice.append([h,w,h+patch_size,w+patch_size])
            n +=1

    return patch_indice

def get_patch_indice_grid(height, width, patch_size,):
    '''
    get patch indices with a grid, areas that are not enough for one patch are discarded

    parameters:
    ------------------
    height: height of the terrain
    width: width of the terrain
    patch_size: patch size

    return:
    ------------------
    list with the shape of [patch_num, 4], the elements are [minH, maxH, minW, maxW]
    '''
    patch_indice = []

    h_num = height // patch_size
    w_num = width // patch_size

    for h in range(h_num):
        for w in range(w_num):
            patch_indice.append([h*patch_size, w*patch_size, (1+h)*patch_size, (1+w)*patch_size])

    return patch_indice

def get_patches(arr, indices, format="HWC"):
    '''
    get patches from terrain array and indice array

    parameters:
    -----------------
    arr: n-d array with shape of [c, h, w] or [h, w, c]
    indices: patch indices
    format: input format, "HWC" for [h, w, c] or "CHW" for [c, h, w]
    '''

    patches=[]
    if len(arr.shape)==2 or format=="HWC":
        for h1,w1,h2,w2 in indices:
            patches.append(arr[h1:h2, w1:w2])
    else:
        for h1,w1,h2,w2 in indices:
            patches.append(arr[:,h1:h2, w1:w2])

    return np.array(patches)

#img, label, rain,  _, _ = load_terrain("data/portugal/terrain.npy","data/portugal/water_level.npy", "data/portugal/rain_pattern.txt", 100, 256)
#print(img.shape)
#print(label.shape)
#print(rain.shape)

#arr=np.load("data/portugal/terrain.npy")

#print(get_patch_indice_random(arr.shape[1],arr.shape[2],5,256))
#print("grid")
#print(get_patch_indice_grid(arr.shape[1],arr.shape[2],256))