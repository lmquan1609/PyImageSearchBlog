from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Dense, Flatten, Input, concatenate
from keras.models import Model
from keras import backend as K

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, k_x, k_y, stride, chan_dim, padding='same'):
        # define a CONV -> ReLU -> BN pattern
        x = Conv2D(K, (k_x, k_y), strides=stride, padding=padding, activation='relu')(x)
        x = BatchNormalization(axis=chan_dim)(x)

        return x
    
    @staticmethod
    def inception_module(x, num_K_1x1, num_K_3x3, chan_dim):
        # define 2 CONV modules, then concatenate across the channel dimension
        conv_1x1 = MiniGoogLeNet.conv_module(x, num_K_1x1, 1, 1, (1, 1), chan_dim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, num_K_3x3, 3, 3, (1, 1), chan_dim)
        x = concatenate([conv_1x1, conv_3x3], axis=chan_dim)

        return x

    @staticmethod
    def downsample_module(x, K, chan_dim):
        # define CONV module and POOL, then concatenate across channel dimension
        conv_3x3 = MiniGoogLeNet.conv_module(x, K, 3, 3, (2, 2), chan_dim, padding='valid')
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chan_dim)

        return x

    @staticmethod
    def build(width, height, depth, classes):
        # initialize input shape to be channels last and channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using channels first, update input shape and channels dimension
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # define model input and first CONV module
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim)

        # 2 Inception modules folowed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chan_dim)
        x = MiniGoogLeNet.downsample_module(x, 80, chan_dim)

        # 4 Inception modules followed by a downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chan_dim)
        x = MiniGoogLeNet.downsample_module(x, 96, chan_dim)

        # 2 Inception modules follwed by global POOL and dropout
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim) # output: (7, 7, 336)
        x = AveragePooling2D((7, 7))(x) # output: (1, 1, 336)
        x = Dropout(0.5)(x)

        # add softmax classifier
        x = Flatten()(x)
        x = Dense(classes, activation='softmax')(x)

        # create model
        model = Model(inputs, outputs=x, name='googlenet')

        return model