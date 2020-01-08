import numpy as np
import math
from typing import List
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file, get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
# For build pip, running out of range of this folder
# from .custom_objects import ConvInitializer, DenseInitializer, Swish, DropConnect
# from .config import BlockArgs, get_default_block_list
# For not build pip, running this file
from custom_objects import ConvInitializer, DenseInitializer, Swish, DropConnect
from config import BlockArgs, get_default_block_list
import os

def round_filters(filters, width_coefficient, depth_divisor, min_depth):
    '''Round number of filters based on depth multiplier'''
    multiplier = float(width_coefficient)
    divisor = int(depth_divisor)
    
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
        x = layers.Lambda(lambda t:
            K.mean(t, axis=spatial_dims, keepdims=True))(x)
        x = layers.Conv2D(num_reduced_filters, (1, 1), 
                            strides=(1, 1),
                            padding='same', 
                            kernel_initializer=ConvInitializer())(x)
        x = Swish()(x)
        # Excitation phase
        x = layers.Conv2D(filters, (1, 1), 
                        strides=(1, 1),
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer=ConvInitializer())(x)
        out = layers.Multiply()([x, inputs]) # Another representation for Swish layer
        return out
    
    return block

def MBConvBlock(input_filters, output_filters,
                kernel_size, strides,
                expand_ratio, se_ratio,
                id_skip, drop_connect_rate,
                batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, 
                data_format=None):
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
            x = layers.Conv2D(filters, (1, 1),
                            strides=(1, 1),
                            padding='same',
                            use_bias=False,
                            kernel_initializer=ConvInitializer())(inputs)
            x = layers.BatchNormalization(axis=chan_dim,
                                    momentum=batch_norm_momentum,
                                    epsilon=batch_norm_epsilon)(x)
            x = Swish()(x)
        else:
            x = inputs

        x = layers.DepthwiseConv2D(kernel_size,
                                strides=strides,
                                padding='same',
                                depthwise_initializer=ConvInitializer(),
                                use_bias=False)(x)
        x = layers.BatchNormalization(axis=chan_dim,
                                momentum=batch_norm_momentum,
                                epsilon=batch_norm_epsilon)(x)
        x = Swish()(x)

        if has_se_layer:
            x = SEBlock(input_filters, se_ratio, expand_ratio)(x)

        x = layers.Conv2D(output_filters, (1, 1),
                        strides=(1, 1),
                        padding='same',
                        use_bias=False,
                        kernel_initializer=ConvInitializer())(x)
        x = layers.BatchNormalization(axis=chan_dim,
                            momentum=batch_norm_momentum,
                            epsilon=batch_norm_epsilon)(x)

        if id_skip:
            if all(s == 1 for s in strides) and (input_filters == output_filters):
                # Only apply drop connect if skip presents
                if drop_connect_rate:
                    x = DropConnect(drop_connect_rate)(x)

                x = layers.Add()([x, inputs])
        
        return x

    return block

def EfficientNet(input_shape,
                block_args_list: List[BlockArgs],
                width_coefficient: float,
                depth_coefficient: float,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000,
                drop_rate=0.,
                drop_connect_rate=0.,
                batch_norm_momentum=0.99,
                batch_norm_epsilon=1e-3,
                depth_divisor=8,
                min_depth=None,
                data_format=None,
                default_size=None,
                **kwargs):
    """
    Builder model for EfficientNets

    # Args:
        input_shape: Optional tuple, depends on the configuration,
            Defaults to 224 when None is provided
        block_args_list: Optional list of BlockArgs,
            each of which detail the args of the MBConvBlock.
            If left as None, it defaults to the blocks from the paper
        width_coefficient: Determine # of channels available per layer
        depth_coefficient: Determine # of layers available to the model
        include_top: Whether to include FC layer at the top of the network
        weights: `None` (random initialization) or `imagenet` (imagenet weights)
            or path to pretrained weight
        input_tensor: optional Keras tensor
        pooling: Optional pooling mode for feature extraction
            when `include_top` is False
            - `None`: the output of the model will be 4D tensor output of
                the last convolutional layer
            - `avg`: global average pooling  will be applied to the output of
                the last convolutional layer, thus its outpus will be 2D tensor
            - `max`: global max pooling  will be applied
        classes: optional # of classes to classify images into,
            only specified if `include_top` is True and `weights` is None
        drop_rate: Float, percentage of dropout
        drop_connect_rate: Float, percentage of random dropped connection
        depth_divisor: Optional. Used when rounding off 
            the coefficient scaled channels and depth of the layers
        min_depth: minimum of depth value to avoid blocks with 0 layer
        default_size: default image size of the model
    # Raises:
        `ValueError`: If weights are not in `imagenet` or None
        `ValueError`: If weights are `imagenet` and `classes` is not 1000
    # Returns:
        A Keras model
    """
    if not (weights in ('imagenet', None) or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                        '`None`, `imagenet` or `path to pretrained weights`')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `imagenet` with `include_top` '
                        'as true, `classes` should be 1000')

    if data_format is None:
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        chan_dim = 1
        spatial_axis = [2, 3]
    else:
        chan_dim = -1
        spatial_axis = [1, 2]

    if default_size is None:
        default_size = 224

    if block_args_list is None:
        block_args_list = get_default_block_list()

    # TODO: count # of strides to compute min size
    stride_count = 1
    for block_args in block_args_list:
        if block_args.strides is not None and block_args.strides[0] > 1:
            stride_count += 1

    min_size = int(2 ** stride_count)
    
    # Determine proper input shape and default size
    input_shape = _obtain_input_shape(input_shape, default_size, min_size,
                                    data_format, include_top, weights=weights)

    # Stem part
    if input_tensor is None:
        inputs = layers.Input(shape=input_shape)
    else:
        if K.is_keras_tensor(input_tensor):
            inputs = input_tensor
        else:
            inputs = layers.Input(shape=input_shape, tensor=input_tensor)

    x = inputs
    # ! parameters in round_filters
    x = layers.Conv2D(
        round_filters(32, width_coefficient, depth_divisor, min_depth),
        (3, 3),
        strides=(2, 2),
        padding='same',
        kernel_initializer=ConvInitializer(),
        use_bias=False)(x)
    x = layers.BatchNormalization(axis=chan_dim,
                                momentum=batch_norm_momentum,
                                epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    num_blocks = sum([block_args.num_repeat for block_args in block_args_list])
    drop_connect_rate_per_block = drop_connect_rate / float(num_blocks)

    # Blocks part
    for block_idx, block_args in enumerate(block_args_list):
        assert block_args.num_repeat > 0, 'Error in # of block'

        # Update block input and output filters based on depth multiplier
        block_args.input_filters = round_filters(
                                    block_args.input_filters,
                                    width_coefficient,
                                    depth_divisor,
                                    min_depth)
        block_args.output_filters = round_filters(
                                        block_args.output_filters,
                                        width_coefficient,
                                        depth_divisor,
                                        min_depth)
        block_args.num_repeat = round_repeats(block_args.num_repeat, depth_coefficient)

        # The first block needs to take care of stride and filter size
        x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                        block_args.kernel_size, block_args.strides,
                        block_args.expand_ratio, block_args.se_ratio,
                        block_args.identity_skip, drop_connect_rate_per_block * block_idx,
                        batch_norm_epsilon, batch_norm_epsilon, data_format)(x)

        if block_args.num_repeat > 1:
            block_args.input_filters = block_args.output_filters
            block_args.strides = (1, 1)

        for _ in range(block_args.num_repeat - 1):
            x = MBConvBlock(block_args.input_filters, block_args.output_filters,
                            block_args.kernel_size, block_args.strides,
                            block_args.expand_ratio, block_args.se_ratio,
                            block_args.identity_skip,
                            drop_connect_rate_per_block * block_idx,
                            batch_norm_epsilon, batch_norm_momentum, data_format)(x)

    # Head part
    x = layers.Conv2D(
        round_filters(1280, width_coefficient, depth_divisor, min_depth),
        (1, 1),
        strides=(1, 1),
        padding='same',
        kernel_initializer=ConvInitializer(),
        use_bias=False
    )(x)
    x = layers.BatchNormalization(axis=chan_dim,
                            momentum=batch_norm_momentum,
                            epsilon=batch_norm_epsilon)(x)
    x = Swish()(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(data_format=data_format)(x)
        if drop_rate > 0:
            x = layers.Dropout(drop_rate)(x)
        x = layers.Dense(classes,
                        activation='softmax',
                        kernel_initializer=DenseInitializer())(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    outputs = x

    # Ensure that the model takes into account any potential predecessors
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs, outputs)

    # Load weights
    if weights == 'imagenet':
        if default_size == 224:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b0.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b0_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b0_notop.h5',
                    cache_subdir='models'
                )

        elif default_size == 240:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b1.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b1_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b1_notop.h5',
                    cache_subdir='models'
                )

        elif default_size == 260:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b2.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b2_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b2_notop.h5',
                    cache_subdir='models'
                )

        elif default_size == 300:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b3.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b3_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b3_notop.h5',
                    cache_subdir='models'
                )

        elif default_size == 380:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b4.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b4_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b4_notop.h5',
                    cache_subdir='models'
                )

        elif default_size == 456:
            if include_top:
                weights_path = get_file(
                    'efficientnet-b5.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5.h5',
                    cache_subdir='models'
                )
            else:
                weights_path = get_file(
                    'efficientnet-b5_notop.h5',
                    'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b5_notop.h5',
                    cache_subdir='models'
                )
        # TODO: Provide links for the last 2 EfficientNet
        # elif default_size == 528:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b6.h5',
        #             'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6.h5',
        #             cache_subdir='models'
        #         )
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b6_notop.h5',
        #             'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b6_notop.h5',
        #             cache_subdir='models'
        #         )

        # elif default_size == 600:
        #     if include_top:
        #         weights_path = get_file(
        #             'efficientnet-b7.h5',
        #             'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7.h5',
        #             cache_subdir='models'
        #         )
        #     else:
        #         weights_path = get_file(
        #             'efficientnet-b7_notop.h5',
        #             'https://github.com/titu1994/keras-efficientnets/releases/download/v0.1/efficientnet-b7_notop.h5',
        #             cache_subdir='models'
        #         )
    elif weights is not None:
        model.load_weights(weights)
    
    return model

def EfficientNetB0(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.2,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=224)

def EfficientNetB1(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.2,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.0,
                        depth_coefficient=1.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=240)

def EfficientNetB2(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.3,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.1,
                        depth_coefficient=1.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=260)

def EfficientNetB3(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.3,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.2,
                        depth_coefficient=1.4,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=300)

def EfficientNetB4(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.4,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.4,
                        depth_coefficient=1.8,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=380)

def EfficientNetB5(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.4,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.6,
                        depth_coefficient=2.2,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=456)

def EfficientNetB6(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.5,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=1.8,
                        depth_coefficient=2.6,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=528)

def EfficientNetB7(input_shape=None,
                    include_top=True,
                    weights='imagenet',
                    input_tensor=None,
                    pooling=None,
                    classes=1000,
                    drop_rate=0.5,
                    drop_connect_rate=0.,
                    data_format=None):
    return EfficientNet(input_shape,
                        get_default_block_list(),
                        width_coefficient=2.0,
                        depth_coefficient=3.1,
                        include_top=include_top,
                        weights=weights,
                        input_tensor=input_tensor,
                        pooling=pooling,
                        classes=classes,
                        drop_rate=drop_rate,
                        drop_connect_rate=drop_connect_rate,
                        data_format=data_format,
                        default_size=600)

if __name__ == '__main__':
    model = EfficientNetB0(include_top=True)
    model.summary()