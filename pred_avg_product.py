
# source: https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson

from keras import backend as K
import os, io , math
import numpy as np
import pandas as pd
import bson
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator
from keras import backend as K
from model_cdis import create_model_vgg16
from tqdm import *
import h5py

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180),
                 with_labels=True, batch_size=32, shuffle=False, seed=None):

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
        self.lock = lock

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

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
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

####################################################################################
save_folder = "submit/"
data_dir="input/"
test_bson_path = "input/test.bson"

num_test_products = 1768182
submission_df = pd.read_csv(data_dir + "sample_submission.csv")
submission_df.head()

categories_df = pd.read_csv("input/categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()

print("creating model...")
model = create_model_vgg16()
print("loading weights...")
model.load_weights('weights/vgg16_0001_idx0_batch1280_zoom0_shift0_adam_31-1.3040-0.7692-0.7317.hdf5', by_name=True)


test_datagen = ImageDataGenerator()
data = bson.decode_file_iter(open(test_bson_path, "rb"))

test_product_score = []

split = 1

with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)

        test_product_score.append(avg_pred)

        if(len(test_product_score) % 1500 == 0):
            print('number of products:{}'.format(len(test_product_score)))
            print('saving up to product_id:{}'.format(product_id))
            print('split:{}'.format(split))
            with h5py.File(save_folder + 'probs-{}.h5'.format(split), 'w') as hf:
                hf.create_dataset("prob-{}".format(split), data=test_product_score)
                del test_product_score
                test_product_score = []
                split = split + 1
        elif(product_id == 23620445):
            print('number of products:{}'.format(len(test_product_score)))
            print('saving up to product_id:{}'.format(product_id))
            print('split:{}'.format(split))
            with h5py.File(save_folder + 'probs-{}.h5'.format(split), 'w') as hf:
                hf.create_dataset("prob-{}".format(split), data=test_product_score)


        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]
        pbar.update()

submission_df.to_csv("incep_res_idxZero_full_flip_Nadam_drop025_32-1.3625-0.7113-0.7036.csv.gz", compression="gzip", index=False)