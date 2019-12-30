import numpy as np
import math
from keras import backend as K
from keras import layers

def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    '''Round number of filters based on depth multiplier'''
    multiplier = float(width_coefficient)
    divisor = int(divisor)
    
    if not multiplier:
        return filter

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    
    return int(new_filters)

def round_repeats(repeats, depth_coefficient):
    multiplier = depth_coefficient

    if not multiplier:
        return repeats
    
    return int(math.ceil(multiplier * repeats))

def SEBlock(input_filters, se_ratio, expand_ratio, data_format=None):
    '''Implement Squeeze and Excitation block'''
    if data_format is None:
        data_format = K.image_data_format()

    num_reduced_filters = max(1, int(input_filters * se_ratio))
    filters = input_filters * expand_ratio

    if data_format == 'channels_first':
        chan_dim = 1
        spatial_dims = [2, 3]
    else:
        chan_dim = -1
        spatial_dims = [1, 2]

    def block(inputs):
        x = inputs
        # Squeeze phase
        x = layers.Lambda(lambda t: K.mean(t, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(num_reduced_filters, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_initializer())(x)
        x = Swish()(x)
        # Excitation phase
        x = layers.Conv2D(filters,  (1, 1), strides=(1, 1), padding='same', activation='sigmoid', kernel_initializer=conv_initializer())(x)
        out = layers.Multiply([x, inputs]) # Another representation for Swish layer
        return out
    
    return block

def MBConvBlock(input_filters, output_filters, kernel_size, strides, expand_ratio, se_ratio, id_skip, drop_connect_rate, batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        chan_dim = 1
        spatial_dims = [2, 3]
    else:
        chan_dim = -1
        spatial_dims = [1, 2]

    has_se_layer = (se_ratio is not None) and (se_ratio > 0 and se_ratio <= 1)
    filters = input_filters * expand_ratio

    def block(inputs):
        if expand_ratio != 1:
            x = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=conv_initializer())(inputs)(x)
            x = layers.BatchNormalization(axis=chan_dim, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', depthwise_initializer=conv_initializer(), use_bias=False)(x)
        x = layers.BatchNormalization(axis=chan_dim, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se_layer:
            x = SEBlock(input_filters, se_ratio, expand_ratio)(x)

        x = layers.Conv2D(output_filters, (1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=conv_initializer())(inputs)(x)
        x = layers.BatchNormalization(axis=chan_dim, momentum=batch_norm_momentum, epsilon=batch_norm_epsilon)(x)

        