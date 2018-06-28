from subprocess import check_output

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

print(check_output(["ls", "input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bson
import os
from tqdm import tqdm

train_folder = 'output/train'
validation_folder = 'output/validation'
test_folder = 'output/test'

# Create train folder
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

# Create validation folder
if not os.path.exists(validation_folder):
    os.makedirs(validation_folder)

# Create test folder
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Create categories folders
categories = pd.read_csv('input/category_names.csv', index_col='category_id')

# for category in tqdm(categories.index):
#    os.mkdir(os.path.join(train_folder, str(category)))
#    os.mkdir(os.path.join(validation_folder, str(category)))

num_products = 1768182  # 7069896 for train and 1768182 for test
num_prod_train = num_products * 0.9  # set 80% of the data as the training set. Leave the remainder as validation set
print('training set will have ', num_prod_train, 'items')

bar = tqdm(total=num_products)
counter = 0
############################################################################################################333
## All images

'''
with open('input/train.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)

    for c, d in enumerate(data):
        category = d['category_id']
        _id = d['_id']
        counter += 1

        for e, pic in enumerate(d['imgs']):
            if counter < num_prod_train:
                fname = os.path.join(train_folder, str(category), '{}-{}.jpg'.format(_id, e))
            else:
                fname = os.path.join(validation_folder, str(category), '{}-{}.jpg'.format(_id, e))
            with open(fname, 'wb') as f:
                f.write(pic['picture'])

        bar.update()
'''
#####################################################################################################################333

## idx_zero
'''
with open('input/train.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)

    for c, d in enumerate(data):
        category = d['category_id']
        _id = d['_id']
        counter += 1

        for e, pic in enumerate(d['imgs']):
            if counter < num_prod_train:
                if e == 0:
                    fname = os.path.join(train_folder, str(category), '{}-{}.jpg'.format(_id, e))
            else:
                if e == 0:
                    fname = os.path.join(validation_folder, str(category), '{}-{}.jpg'.format(_id, e))
            with open(fname, 'wb') as f:
                f.write(pic['picture'])

        bar.update()
'''
#######################################################################################################################3

## Test images

with open('input/test.bson', 'rb') as fbson:
    data = bson.decode_file_iter(fbson)

    for c, d in enumerate(data):
        # category = d['category_id']
        _id = d['_id']
        # counter += 1
        for e, pic in enumerate(d['imgs']):
            fname = os.path.join(test_folder, '{}-{}.jpg'.format(_id, e))
            with open(fname, 'wb') as f:
                f.write(pic['picture'])

        bar.update()
