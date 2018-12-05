# urban_flood
urban water flood prediction using machine learning

![](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/data/images/test%20tr10-2.png)
*result on a testing rain pattern*

![](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/data/images/rnd_15.png)
*result on a random generated rain pattern*

## code
the latest file for portugal is [urban_flood/model_portugal/shuffle_patterns.py (training and testing)](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/model_portugal/shuffle_patterns.py) and [urban_flood/interact.ipynb (jupyter interactive)](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/interact.ipynb).


the model for the latest result is [urban_flood/data/model/portugal_shuffle_pattern](https://github.com/guozifeng91/urban_flood/tree/master/urban_flood/data/model/portugal_shuffle_pattern), which is trained using random patches and 3/4 of the rain patterns. 1/4 of the patterns are test set. the code for the model is [urban_flood/model_portugal/nn_model.py](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/model_portugal/nn_model.py)

## data
data for luzern and hoengg are in the folder [data/luzern](https://github.com/guozifeng91/urban_flood/tree/master/urban_flood/data/luzern) and [data/hoengg](https://github.com/guozifeng91/urban_flood/tree/master/urban_flood/data/hoengg), as original csv. data for portugal is in [data/portugal](https://github.com/guozifeng91/urban_flood/tree/master/urban_flood/data/portugal), the format are numpy .npy and are compressed for uploading. the code for preprocessing the portugal data is [urban_flood/data process.ipynb](https://github.com/guozifeng91/urban_flood/blob/master/urban_flood/data%20process.ipynb) 

## result
the latest results are in [urban_flood/data/images](https://github.com/guozifeng91/urban_flood/tree/master/urban_flood/data/images), other folder start with "image" are the results for previous experiments
