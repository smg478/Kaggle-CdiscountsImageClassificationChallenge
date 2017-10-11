import os, sys, math, io
import numpy as np
import pandas as pd
import bson
import pickle
import gzip
import cv2
from multiprocessing import cpu_count
from tqdm import *

from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from model_cdis import create_model_vgg16


##############################################################################
class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, target_size=(180, 180), with_labels=True,
                 batch_size=32, shuffle=False, seed=None):

        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)

        print("Found %d images belonging to %d classes." % (self.samples, self.num_class))

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)


def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


##############################################################################
num_classes = 5270
batch_size = 128
num_test_products = 1768182  # products which contains multiple images
# Found 3095080 images belonging to 5270 classes.

data_dir = "input/"
test_bson_path = os.path.join(data_dir, "test.bson")
test_bson_file = open(test_bson_path, "rb")

test_offsets_df = pd.read_csv("input/test_offsets.csv", index_col=0)
test_images_df = pd.read_csv("input/test_images.csv", index_col=0)
categories_df = pd.read_csv("input/categories.csv", index_col=0)

cat2idx, idx2cat = make_category_tables()

print("creating model...")
model = create_model_vgg16()
print("loading weights...")
model.load_weights(filepath='weights/weight_vgg16_f10_Aug.00-1.62117-0.75646-0.67416.hdf5')
##############################################################################
# split dataset
test_images_df_1 = test_images_df[:309504]
test_images_df_2 = test_images_df[309504:619012]
test_images_df_3 = test_images_df[619012:928514]
test_images_df_4 = test_images_df[928514:1238022]
test_images_df_5 = test_images_df[1238022:1547530]
test_images_df_6 = test_images_df[1547530:1857023]
test_images_df_7 = test_images_df[1857023:2166540]
test_images_df_8 = test_images_df[2166540:2476062]
test_images_df_9 = test_images_df[2476062:2785564]
test_images_df_10 = test_images_df[2785564:3095080]

test_im_batch = (test_images_df_1, test_images_df_2, test_images_df_3, test_images_df_4, test_images_df_5,
                 test_images_df_6, test_images_df_7, test_images_df_8, test_images_df_9, test_images_df_10)

####################################################################
split = 1
y_full_test = []
p_full_test = []
tta_times = 5

if 1:
    for im in tqdm(test_im_batch):
        test_images_df_new = im
        print("predicting from split {}...".format(split))
        print("No. of images in split %d..." % len(test_images_df_new))
        ######################3#################################################################
        for i in range(tta_times):
            test_datagen = ImageDataGenerator(width_shift_range=0.1,
                                              height_shift_range=0.1,
                                              horizontal_flip=True,
                                              vertical_flip=False,
                                              shear_range=0.0,
                                              zoom_range=0.0,
                                              fill_mode='reflect')
            test_gen = BSONIterator(test_bson_file, test_images_df_new, test_offsets_df,
                                    num_classes, test_datagen, batch_size=batch_size,
                                    with_labels=False, shuffle=False)

            print("predict type = %d" % i)
            p_test = model.predict_generator(test_gen,
                                             steps=np.ceil(float(len(test_images_df_new)) / float(batch_size)),
                                             # steps=1000,
                                             workers=6)
            p_full_test.append(p_test)

        p_test = np.array(p_full_test[0])
        for i in range(1, tta_times):
            p_test += np.array(p_full_test[i])
        p_test /= tta_times

        ##################################################################################3#########
        cat_idx = np.argmax(p_test, axis=1)
        cat_idx = np.array([cat_idx])
        cat_idx = cat_idx.T
        test_images_np = test_images_df_new.values
        test_pred = np.concatenate(([test_images_np, cat_idx]), axis=1)
        # keep only relevant column
        test_pred = np.delete(test_pred, 1, 1)
        # convert to dataframe
        columns = ['_id', 'category']
        dtype = [('_id', 'int32'), ('category', 'int32')]
        test_pred_df = pd.DataFrame({'_id': test_pred[:, 0], 'category': test_pred[:, 1]})
        test_pred_df.to_csv('submit/raw_results_{}.csv.gz'.format(split), index=False, compression='gzip')

        del test_images_df_new, p_test
        split = split + 1

# Read result segments from csv
file_names = ['submit/raw_results_1.csv.gz', 'submit/raw_results_2.csv.gz', 'submit/raw_results_3.csv.gz',
              'submit/raw_results_4.csv.gz',
              'submit/raw_results_5.csv.gz', 'submit/raw_results_6.csv.gz', 'submit/raw_results_7.csv.gz',
              'submit/raw_results_8.csv.gz',
              'submit/raw_results_9.csv.gz', 'submit/raw_results_10.csv.gz']
results = pd.DataFrame()
if 1:
    for filename in file_names:
        results = results.append(pd.read_csv(filename), ignore_index=True)

grouped = results.groupby('_id')
xx = grouped['category'].apply(lambda x: x.mode()[0])
yy = pd.DataFrame({'_id': xx.index, 'category_id': xx.values})

for index, rows in yy.iterrows():
    rows['category_id'] = idx2cat[rows['category_id']]

print("Generating submission file...")
yy.to_csv('submit/weight_vgg16_f10_Aug.00-1.62117-0.75646-0.67416_TTA5.csv.gz', index=False, compression='gzip')
