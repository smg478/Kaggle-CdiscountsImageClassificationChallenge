from keras import applications
from keras import layers
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def create_model_vgg16():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.VGG16(weights=None, include_top=False, input_tensor=input_tensor)  # 224x224

    y = base_model.output
    y = Flatten()(y)
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:10]:
            layer.trainable = True
        for layer in model.layers[10:]:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_xception():
    input_tensor = Input(shape=(180, 180, 3))
    base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)

    y = base_model.output
    y = GlobalAveragePooling2D(name='avg_pool')(y)
    y = Dropout(0.2)(y)

    predictions = Dense(5270, activation='softmax', name='predictions')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:85]:
            layer.trainable = True
        for layer in model.layers[85:]:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_incep_res():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D()(y)

    y = Dense(2048)(y)
    y = Dropout(0.25)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)

    y = Dense(2048)(y)
    y = Dropout(0.25)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)

    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:400]:
            layer.trainable = True
        for layer in model.layers[400:]:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_res50():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 224x224

    y = base_model.output
    y = Flatten()(y)
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:152]:
            layer.trainable = False
        for layer in model.layers[152:]:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_incepv3():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    y = Dense(3000,name='fc1')(y)
    y = Dropout(0.2)(y)
    predictions = Dense(5270, activation='softmax', name='predictions')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:151]:
            layer.trainable = True
        for layer in model.layers[151:]:
            layer.trainable = True

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model
