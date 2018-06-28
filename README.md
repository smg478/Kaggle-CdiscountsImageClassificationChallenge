## Cdiscount-2017
This repository contains the keras solution files of the challenge.

## Requirements

Keras 2.0.8 w/ TF backend
sklearn
cv2
tqdm
h5py

## Usage

### Data preparation

Follow [Human Analog's](https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson) kernel for data preparation. Place all items in to input folder.

### Train

Run python train.py to train the model using pre-trained weights.

models_cdis.py contains all the model architecture definitions.

### Test and submit

Place trained weights in to the 'weights' folder (assumed to be under base directory).

Run python submit.py to make average predictions of a product which contains 1-4 images.

Run python ensemble.py to make most common label ensemble from submission files.


### Tools

'dev_files' folder contains some useful codes for experiment and result analysis.

