from keras import applications
from keras.models import Model, load_model
from keras import layers
from keras.optimizers import SGD, Adagrad, Adam, Nadam
from keras.layers import Input, Activation, merge, Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D, \
    AveragePooling2D
from keras.layers.normalization import BatchNormalization


def create_model_template():
    input_tensor = Input(shape=(180, 180, 3))

    # create the base pre-trained model
    # base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)    #299x299
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 224x224
    # base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)       #224x224
    # base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)        #299x299
    # base_model = applications.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input_tensor)  #299x299

    y = base_model.output

    ################# Vgg16, vgg19
    y = Flatten()(y)  # Res50, vgg16, vgg19
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    ############## inception-resnet
    # y = GlobalAveragePooling2D()(y)       # xception,inception, insection-resnet
    # y = Dense(2048)(y)
    # y = layers.advanced_activations.LeakyReLU(0.2)(y)
    # y = BatchNormalization()(y)
    # predictions = Dense(5270, activation='softmax')(y)


    # x = layers.Dense(1024)(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.advanced_activations.LeakyReLU()(x)
    # x = layers.Dropout(0.25)(x)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    model.load_weights('weights/weight_vgg16_f10_Aug.01-1.81250-0.61828-0.63671.hdf5', by_name=True)
    print('Weights loaded.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151, inception-resnet 400
            layer.trainable = False
    else:
        for layer in model.layers[:10]:
            layer.trainable = False
        for layer in model.layers[10:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_vgg16():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.VGG16(weights=None, include_top=False, input_tensor=input_tensor)  # 224x224

    y = base_model.output
    y = Flatten()(y)  # Res50, vgg16, vgg19
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
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
            layer.trainable = False
    else:
        for layer in model.layers[:10]:
            layer.trainable = False
        for layer in model.layers[10:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=["accuracy"])  ##1e4
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_xception():
    input_tensor = Input(shape=(180, 180, 3))
    base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D()(y)  # xception, inception, insection-resnet
    y = Dense(1024)(y)
    y = layers.advanced_activations.ELU()(y)
    y = BatchNormalization()(y)
    y = Dense(1024)(y)
    y = layers.advanced_activations.ELU()(y)
    y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
            layer.trainable = False
    else:
        for layer in model.layers[:85]:
            layer.trainable = False
        for layer in model.layers[85:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_incep_res():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D()(y)  # xception, inception, insection-resnet
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    # y = Dense(2048)(y)
    # y = layers.advanced_activations.LeakyReLU(0.2)(y)
    # y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
            layer.trainable = False
    else:
        for layer in model.layers[:85]:
            layer.trainable = False
        for layer in model.layers[85:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    # model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_res50():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 224x224

    y = base_model.output
    y = Flatten()(y)  # Res50, vgg16, vgg19
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
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
            layer.trainable = False
    else:
        for layer in model.layers[:152]:
            layer.trainable = False
        for layer in model.layers[152:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    # model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_incepv3():
    input_tensor = Input(shape=(180, 180, 3))

    base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D()(y)  # xception,inception, insection-resnet
    y = Dense(2048)(y)
    y = layers.advanced_activations.LeakyReLU(0.2)(y)
    y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151, inception-resnet 400
            layer.trainable = False
    else:
        for layer in model.layers[:151]:
            layer.trainable = False
        for layer in model.layers[151:]:
            layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=["accuracy"])  ##1e4
    print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model
