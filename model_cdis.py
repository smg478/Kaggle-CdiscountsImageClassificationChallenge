from keras import applications
from keras.models import Model, load_model
from keras import layers
from keras.optimizers import SGD, Adagrad, Adam, Nadam
from keras.layers import Input, Activation, merge, Dense, Flatten, GlobalAveragePooling2D, Dropout, Conv2D
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K

from custom_layers.scale_layer import Scale


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
            layer.trainable = True
        for layer in model.layers[10:]:
            layer.trainable = True

    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001), metrics=["accuracy"])  ##1e4
    #print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model


def create_model_xception():
    input_tensor = Input(shape=(180, 180, 3))
    base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)  # 299x299

    y = base_model.output
    y = GlobalAveragePooling2D(name='avg_pool')(y)  # xception, inception, insection-resnet
    y = Dropout(0.2)(y)
    #y = Dense(2048,name='dense1')(y)
    #y = layers.advanced_activations.ELU()(y)
    #y = BatchNormalization()(y)
    # y = Dense(1024)(y)
    # y = layers.advanced_activations.ELU()(y)
    # y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax', name='predictions')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151
            layer.trainable = False
    else:
        for layer in model.layers[:85]:
            layer.trainable = True
        for layer in model.layers[85:]:
            layer.trainable = True

    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=["accuracy"])  ##1e4
    #print('Model loaded.')

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
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151 in-res 789
            layer.trainable = False
    else:
        for layer in model.layers[:400]:
            layer.trainable = True
        for layer in model.layers[400:]:
            layer.trainable = True


    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.000005), metrics=["accuracy"])  ##1e4
    #model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    #print('Model loaded.')

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
    y = GlobalAveragePooling2D()(y)  # xception, inception, insection-resnet
    y = Dense(3000,name='fc1')(y)
    y = Dropout(0.2)(y)
    #y = layers.advanced_activations.ELU()(y)
    #y = BatchNormalization()(y)
    # y = Dense(1024)(y)
    # y = layers.advanced_activations.ELU()(y)
    # y = BatchNormalization()(y)
    predictions = Dense(5270, activation='softmax', name='predictions')(y)

    model = Model(input=input_tensor, output=predictions)
    print('Model created.')

    train_top_only = False

    if train_top_only:
        for layer in base_model.layers:  # vgg19 11,17 layer freeze, Res50 91,140,152,162,172 layer freeze, xception 85, inception 151, inception-resnet 400
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


def create_model_res101(img_rows, img_cols, color_type=1, num_classes=None):

    def identity_block(input_tensor, kernel_size, filters, stage, block):
        '''The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        '''
        eps = 1.1e-5
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
        x = Activation('relu', name=conv_name_base + '2a_relu')(x)

        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
        x = Conv2D(nb_filter2, kernel_size, kernel_size,
                          name=conv_name_base + '2b', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b_relu')(x)

        x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

        x = merge([x, input_tensor], mode='sum', name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x

    def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        '''conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well
        '''
        eps = 1.1e-5
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        scale_name_base = 'scale' + str(stage) + block + '_branch'

        x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                          name=conv_name_base + '2a', bias=False)(input_tensor)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
        x = Activation('relu', name=conv_name_base + '2a_relu')(x)

        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
        x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                          name=conv_name_base + '2b', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
        x = Activation('relu', name=conv_name_base + '2b_relu')(x)

        x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
        x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
        x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

        shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                                 name=conv_name_base + '1', bias=False)(input_tensor)
        shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
        shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

        x = merge([x, shortcut], mode='sum', name='res' + str(stage) + block)
        x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
        return x


    """
    Resnet 101 Model for Keras
    Model Schema and layer naming follow that of the original Caffe implementation
    https://github.com/KaimingHe/deep-residual-networks
    ImageNet Pretrained Weights
    Theano: https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing
    TensorFlow: https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 4):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


    #x_fc = AveragePooling2D((4, 4), name='avg_pool')(x)
    #x_fc = Flatten()(x_fc)
    #x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    #model = Model(img_input, x_fc)

    #weights_path = 'imagenet_models/resnet101_weights_tf.h5'

    #model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    #x_newfc = GlobalAveragePooling2D()(x)
    x_newfc = AveragePooling2D((5, 5), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    for layer in model.layers[:400]:
        layer.trainable = True
    for layer in model.layers[400:]:
        layer.trainable = True

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.000005), metrics=["accuracy"])  ##1e4
    # model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    # print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    # Learning rate is changed to 0.001
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_model_densenet121(img_rows, img_cols, color_type=1, nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.5, dropout_rate=0.0, weight_decay=1e-4, num_classes=None):

    def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
        '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                branch: layer index within each dense block
                nb_filter: number of filters
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''
        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_' + str(branch)
        relu_name_base = 'relu' + str(stage) + '_' + str(branch)

        # 1x1 Convolution (Bottleneck layer)
        inter_channel = nb_filter * 4
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x1')(x)
        x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # 3x3 Convolution
        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
        x = Activation('relu', name=relu_name_base+'_x2')(x)
        x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
        x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
        ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_filter: number of filters
                compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
                dropout_rate: dropout rate
                weight_decay: weight decay factor
        '''

        eps = 1.1e-5
        conv_name_base = 'conv' + str(stage) + '_blk'
        relu_name_base = 'relu' + str(stage) + '_blk'
        pool_name_base = 'pool' + str(stage)

        x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
        x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
        x = Activation('relu', name=relu_name_base)(x)
        x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

        return x

    def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
        ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
        '''

        eps = 1.1e-5
        concat_feat = x

        for i in range(nb_layers):
            branch = i+1
            x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
            concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

            if grow_nb_filters:
                nb_filter += growth_rate

        return concat_feat, nb_filter

    '''
    DenseNet 121 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6,12,24,16] # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
      # Use pre-trained weights for Theano backend
      weights_path = 'imagenet_models/densenet121_weights_th.h5'
    else:
      # Use pre-trained weights for Tensorflow backend
      weights_path = 'imagenet_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    for layer in model.layers[:100]:
        layer.trainable = False
    for layer in model.layers[100:]:
        layer.trainable = True

    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.000005), metrics=["accuracy"])  ##1e4
    # model.compile(loss='categorical_crossentropy', optimizer = Nadam(), metrics=["accuracy"])
    # print('Model loaded.')

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    model.summary()

    return model
