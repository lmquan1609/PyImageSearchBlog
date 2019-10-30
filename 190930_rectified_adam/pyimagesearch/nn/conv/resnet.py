from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Activation, Dense, Flatten, add
from keras.models import Input, Model
from keras.regularizers import l2
from keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chan_dim, red=False, reg=1e-4, bn_eps=2e-5, bn_mom=0.9):
        # define shortcut which is used for identity mapping
        shortcut = data
        
        # first block of the ResNet module are the 1x1 Convs
        bn1 = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(K*0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # second block of the ResNet module are the 3x3 convs
        bn2 = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K*0.25), (3, 3), strides=stride, padding='same', use_bias=False, kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1 Convs
        bn3 = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if reducing spatial size, apply a Conv layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together to shortcut and final Conv
        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4, bn_eps=2e-5, bn_mom=0.9, dataset='cifar'):
        # initialize input shape to be channels last and channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using channels first, update input shape and channels dimension
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1
        
        # set input and apply BN
        inputs = Input(shape = input_shape)
        x = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(inputs)

        # check if using CIFAR10
        if dataset == 'cifar':
            # apply single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)

        # check to see if using Tiny Imagenet dataset
        elif dataset == 'tiny_imagenet':
            # apply CONV -> ACT -. BN -> POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias=False, padding='same', activation='relu', kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(x)
            x = ZeroPadding2D()(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over then number of stages
        for i in range(len(stages)):
            # initialize the stride, then apply a residual module used to reduce spatial size of input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], stride, chan_dim, red=True, bn_eps=bn_eps, bn_mom=bn_mom)

            # loop over the number of layers in the stage
            for j in range(stages[i] - 1):
                # apply ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chan_dim, bn_eps=bn_eps, bn_mom=bn_mom)

        # apply BN -> ACT -> POOL
        x = BatchNormalization(axis=chan_dim, momentum=bn_mom, epsilon=bn_eps)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(reg))(x)

        # create model
        model = Model(inputs, x, name='resnet')

        return model