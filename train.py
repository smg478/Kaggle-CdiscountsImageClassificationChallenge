import io
import math
import os
import random
import threading
import warnings

import bson
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
# from model_cdis import create_model_vgg16
# from duel_path_network import *
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import load_img, img_to_array, Iterator, ImageDataGenerator

# from model_cdis import create_model_incep_res
# from model_cdis import create_model_xception
# from model_cdis import create_model_res101
from model_cdis import create_model_incepv3

np.random.seed(2016)
random.seed(2016)
warnings.filterwarnings("ignore")

lock = threading.Lock()


# https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
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


model_name = "incep3"
models_savename = "weights/" + model_name
data_dir = "input/"
train_bson_path = os.path.join(data_dir, "train.bson")
train_bson_file = open(train_bson_path, "rb")
categories_df = pd.read_csv("input/categories.csv", index_col=0)
train_offsets_df = pd.read_csv("input/train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("input/train_images.csv", index_col=0)
val_images_df = pd.read_csv("input/val_idx_zero.csv", index_col=0)

num_classes = 5270
batch_size = 96

cat2idx, idx2cat = make_category_tables()
print('train images:', len(train_images_df))
print('Val images:', len(val_images_df))


def run_train():
    callbacks = [ModelCheckpoint(monitor='val_loss',
                                 filepath=models_savename + '_00005_batch1880_zoom0_shift0_adam_{epoch:02d}-{val_loss:.4f}-{acc:.4f}-{val_acc:.4f}.hdf5',
                                 save_best_only=False,
                                 save_weights_only=True,
                                 period=1),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=np.sqrt(0.1),
                                   patience=4,
                                   cooldown=1,
                                   verbose=1,
                                   epsilon=1e-4),
                 TensorBoard(log_dir='logs/{}'.format(model_name))]

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       featurewise_center=False,
                                       samplewise_center=False,
                                       featurewise_std_normalization=False,
                                       samplewise_std_normalization=False,
                                       zca_whitening=False,
                                       rotation_range=0,
                                       width_shift_range=0.0,
                                       height_shift_range=0.0,
                                       horizontal_flip=False,
                                       vertical_flip=False,
                                       shear_range=0.0,
                                       zoom_range=0.0,
                                       fill_mode='constant',
                                       cval=0.)

    train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                             num_classes, train_datagen, lock,
                             batch_size=batch_size, shuffle=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           num_classes, val_datagen, lock,
                           batch_size=batch_size, shuffle=True)

    next(train_gen)  # warm-up

    model = create_model_incepv3()
    model.load_weights('weights/incep3_0001_idx0_batch960_zoom0_shift0_adam_53-1.1159-0.8072-0.7309.hdf5', by_name=True)
    # model = load_model('weights/inception_v2.hdf5')
    print('Weights loaded.')

    init_epoch_arr = [0, 5, 10, 13, 16, 19, 21, 23]
    epochs_arr = [5, 10, 13, 16, 19, 21, 23, 25]
    learn_rates = [0.0000085, 0.0000070, 0.0000055, 0.0000040, 0.0000025, 0.000001, 0.00000070, 0.0000005]

    for learn_rate, epochs, init_epoch in zip(learn_rates, epochs_arr, init_epoch_arr):
        print('learning rate = {}'.format(learn_rate))
        opt = Adam(lr=learn_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model.fit_generator(train_gen,
                            initial_epoch=init_epoch,
                            steps_per_epoch=math.ceil(1000000 / batch_size),
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=val_gen,
                            validation_steps=math.ceil(500000 / batch_size),
                            workers=8,
                            max_queue_size=10,
                            # class_weight=class_weight2
                            )


# To evaluate on the validation set:
def run_evaluation():
    model = create_model_incepv3()
    model.load_weights('weights/incep3_0001_idx0_batch960_zoom0_shift0_adam_53-1.1159-0.8072-0.7309.hdf5', by_name=True)
    print('Weights loaded.')
    opt = Nadam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                           num_classes, val_datagen, lock,
                           batch_size=batch_size, shuffle=True)

    print('evaluating model ...')
    x = model.evaluate_generator(val_gen, steps=np.ceil(50000 / batch_size), workers=8)
    print(x)
    print(model.metrics_names)
    print('done')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_train()
    # run_evaluation()
    print('\nsucess!')
