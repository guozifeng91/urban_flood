# urban_flood
fast urban flood prediction using convolutional neural network

a convolutional neural network is trained to predict the maximum water depth of a specific catchment area by the input rainfall pattern (hyetographs)

the convolutional neural network merge the information of the terrain and rainfall pattern in its latent layer, and predict the corresponding water depth in the output layer

the result of the entire catchment area is assembled from small patches

![](https://github.com/guozifeng91/urban_flood/blob/master/images/pipeline.jpg)
*pipeline*

![](https://github.com/guozifeng91/urban_flood/blob/master/images/model.jpg)
*prediction model*

![](https://github.com/guozifeng91/urban_flood/blob/master/images/accuracy.jpg)
*accuracy*

![](https://github.com/guozifeng91/urban_flood/blob/master/images/prediction_simulation.jpg)
*result on a testing rain pattern*

![](https://github.com/guozifeng91/urban_flood/blob/master/images/gen_rain.jpg)
*result on a random generated rain pattern*

![](https://github.com/guozifeng91/urban_flood/blob/master/images/enlargement.jpg)
*analysis of high-error area*

## code and data
The source code and the training data for the project are hosted by [ETH recearch collection](https://www.research-collection.ethz.ch/handle/20.500.11850/365484)

## paper
the data is accociated with [this paper](https://arxiv.org/abs/2004.08340)
