import io
import os

import bson
import h5py
import numpy as np
import pandas as pd
from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import load_img, img_to_array
from tqdm import *


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


##############################################################################

num_classes = 5270
batch_size = 512
num_test_products = 1768182  # products which contains multiple images
# Found 3095080 images belonging to 5270 classes.

data_dir = "input/"
test_bson_path = os.path.join(data_dir, "test.bson")
test_bson_file = open(test_bson_path, "rb")

test_offsets_df = pd.read_csv("input/test_offsets.csv", index_col=0)
test_images_df = pd.read_csv("input/test_idx_zero.csv", index_col=0)
categories_df = pd.read_csv("input/categories.csv", index_col=0)

cat2idx, idx2cat = make_category_tables()


##############################################################################

def run_merge():
    # split dataset
    test_images_df_1 = test_images_df[:150000]
    test_images_df_2 = test_images_df[150000:300000]
    test_images_df_3 = test_images_df[300000:450000]
    test_images_df_4 = test_images_df[450000:600000]
    test_images_df_5 = test_images_df[600000:750000]
    test_images_df_6 = test_images_df[750000:900000]
    test_images_df_7 = test_images_df[900000:1050000]
    test_images_df_8 = test_images_df[1050000:1200000]
    test_images_df_9 = test_images_df[1200000:1350000]
    test_images_df_10 = test_images_df[1350000:1500000]
    test_images_df_11 = test_images_df[1500000:1650000]
    test_images_df_12 = test_images_df[1650000:1768182]

    test_im_batch = (test_images_df_1,
                     test_images_df_2,
                     test_images_df_3,
                     test_images_df_4,
                     test_images_df_5,
                     test_images_df_6,
                     test_images_df_7,
                     test_images_df_8,
                     test_images_df_9,
                     test_images_df_10,
                     test_images_df_11,
                     test_images_df_12
                     )

    #########################################################################################################
    split = 1

    for im in tqdm(test_im_batch):
        test_images_df_new = im

        print("predicting from split {}...".format(split))
        print("No. of images in split %d..." % len(test_images_df_new))

        # https://stackoverflow.com/questions/28170623/how-to-read-hdf5-files-in-python
        filename = '/media/galib/Documents/cdiscount/submit/xception_idx0/probs-{}.h5'.format(split)
        f = h5py.File(filename, 'r')
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        data_1 = list(f[a_group_key])
        data_1 = np.array(data_1)
        print("data_1 loaded.")

        filename = '/media/galib/Documents/cdiscount/submit/incep_res_idx0/probs-{}.h5'.format(split)
        f = h5py.File(filename, 'r')
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        data_2 = list(f[a_group_key])
        data_2 = np.array(data_2)
        print("data_2 loaded.")

        filename = '/media/galib/Documents/cdiscount/submit/incep3_idx0/probs-{}.h5'.format(split)
        f = h5py.File(filename, 'r')
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        data_3 = list(f[a_group_key])
        data_3 = np.array(data_3)
        print("data_3 loaded.")

        print("averaging scores...")
        avg_score = (data_1 + data_2 + data_3) / 3
        print("averaging done.")
        ######### Save probs

        with h5py.File(save_folder + 'probs-{}.h5'.format(split), 'w') as hf:
            hf.create_dataset("prob-{}".format(split), data=avg_score)
        print("avg scores saved")
        ##################################################################################3#########
        cat_idx = np.argmax(avg_score, axis=1)
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
        test_pred_df.to_csv(save_folder + 'raw_results_{}.csv.gz'.format(split), index=False, compression='gzip')

        del test_images_df_new, data_1, data_2, data_3, avg_score
        split = split + 1


def read_raw_submit():
    # Read result segments from csv
    print('reading raw files ...')
    file_names = [save_folder + 'raw_results_1.csv.gz',
                  save_folder + 'raw_results_2.csv.gz',
                  save_folder + 'raw_results_3.csv.gz',
                  save_folder + 'raw_results_4.csv.gz',
                  save_folder + 'raw_results_5.csv.gz',
                  save_folder + 'raw_results_6.csv.gz',
                  save_folder + 'raw_results_7.csv.gz',
                  save_folder + 'raw_results_8.csv.gz',
                  save_folder + 'raw_results_9.csv.gz',
                  save_folder + 'raw_results_10.csv.gz',
                  save_folder + 'raw_results_11.csv.gz',
                  save_folder + 'raw_results_12.csv.gz',
                  ]

    print('raw files loaded.')
    results = pd.DataFrame()

    for filename in file_names:
        results = results.append(pd.read_csv(filename), ignore_index=True)

    grouped = results.groupby('_id')
    xx = grouped['category'].apply(lambda x: x.mode()[0])
    yy = pd.DataFrame({'_id': xx.index, 'category_id': xx.values})

    for index, rows in yy.iterrows():
        rows['category_id'] = idx2cat[rows['category_id']]

    print("Generating submission file...")
    yy.to_csv(save_folder + save_name + '.csv.gz', index=False, compression='gzip')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    save_folder = 'submit/merge_incep_incepRes_xcep_idx0/'
    save_name = 'incep_incepRes_xcep_avgScore'

    run_merge()
    read_raw_submit()

    print('\nsucess!')
