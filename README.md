## Cdiscount-2017
This repository contains the keras solution files of the challange. 

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

Run python train_cdiscount.py to train the model. 

models_cdis.py contains all the model architecture definitions.

### Test and submit

Place trained weights in to the 'weights' folder.

Run python pred_avg_product.py to make average predictions of a product which contains 1-4 images.

Run python ensemble_from_submission.py to make most common label ensemble from submission files.

Run python TTA_cdis.py to do test time augmentation and make pridictions. This script doesn't make average prediction of a product. Instead, this treats every image as an individual product. Then makes most common label prediction on a product. (If there are more than one same frequent label, then it takes the value which comes first) - not a perfect prediction strategy.

### Tools

'tools' folder contains some useful codes for experiment and result analysis.

