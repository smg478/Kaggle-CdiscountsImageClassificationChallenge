import os, io
import warnings
import random
import numpy as np
import pandas as pd
import bson
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import backend as K
from model_cdis import create_model_vgg16

warnings.filterwarnings("ignore")

np.random.seed(2016)
random.seed(2016)


#########################################################################################################

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


############################################################################################################3

data_dir = "input/"

train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896

categories_df = pd.read_csv("input/categories.csv", index_col=0)
cat2idx, idx2cat = make_category_tables()

train_offsets_df = pd.read_csv("input/train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("input/train_images_50_2.csv", index_col=0)
val_images_df = pd.read_csv("input/val_images.csv", index_col=0)

print('train images:', len(train_images_df))
print('Val images:', len(val_images_df))

train_bson_file = open(train_bson_path, "rb")
num_classes = 5270
batch_size = 128

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=3,
                           verbose=2,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               patience=2,
                               cooldown=2,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='weights/weight_vgg16_freeze10_shift_flip_zoom_lr_00005.{epoch:02d}-{val_loss:.5f}-{acc:.5f}-{val_acc:.5f}.hdf5',
                             save_best_only=False,
                             save_weights_only=True,
                             period=1),
             TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=True)]

train_datagen = ImageDataGenerator(  # rescale=1. /255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    shear_range=0.0,
    zoom_range=0.1,
    fill_mode='reflect')

train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                         num_classes, train_datagen, batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, batch_size=batch_size)

model = create_model_vgg16()
model.load_weights('weights/weight_vgg16_f10_Aug.00-1.62117-0.75646-0.67416.hdf5', by_name=True)
print('Weights loaded.')

model.fit_generator(train_gen,
                    # steps_per_epoch=100,
                    steps_per_epoch=(np.ceil(float(len(train_images_df)) / float(batch_size))),
                    epochs=10,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    validation_steps=np.ceil(float(len(val_images_df)) / float(batch_size)),
                    workers=6)
# validation_steps=100)

# To evaluate on the validation set:
# model.evaluate_generator(val_gen, steps=np.ceil(float(len(val_images_df)) / float(batch_size)), workers=8)
