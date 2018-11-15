import numpy as np

'''
load npy data for portugal case
'''

def load_terrain_grid(path_terrain, path_label, path_pattern, patch_size, dtype_img=np.float32, dtype_label=np.float32):
    '''
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

#img, label, rain,  _, _ = load_terrain("data/portugal/terrain.npy","data/portugal/water_level.npy", "data/portugal/rain_pattern.txt", 100, 256)
#print(img.shape)
#print(label.shape)
#print(rain.shape)