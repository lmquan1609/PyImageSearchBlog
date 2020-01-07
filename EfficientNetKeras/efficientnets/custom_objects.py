import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, layers
from keras.utils.generic_utils import get_custom_objects

class ConvInitializer(initializers.Initializer):
    def __init__(self):
        super().__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, output_filters = shape
        fan_out = int(kernel_height * kernel_width * output_filters)
        return K.random_normal(shape,
                            mean=0.0,
                            stddev=np.sqrt(2.0/fan_out),
                            dtype=dtype)

class DenseInitializer(initializers.Initializer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return K.random_uniform(shape,
                                minval=-init_range,
                                maxval=init_range,
                                dtype=dtype)

class Swish(layers.Layer)                                :
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.support_masking = True

    def call(self, inputs, training=None):
        return tf.nn.swish(inputs)

class DropConnect(layers.Layer)        :
    def __init__(self, drop_connect_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_connect_rate = float(drop_connect_rate)

    def call(self, inputs, training=None):
        def drop_connect():
            keep_prob = 1.0 - self.drop_connect_rate

            # Compute drop_connect tensor
            batch_size = tf.shape(inputs)[0]
            random_tensor = keep_prob
            random_tensor += K.random_uniform((batch_size, 1, 1, 1), dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output

        return K.in_train_phase(drop_connect, inputs, training=training)

    def get_config(self):
        config = {
            'drop_connect_rate': self.drop_connect_rate
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

get_custom_objects().update({
    'ConvInitializer': ConvInitializer,
    'DenseInitializer': DenseInitializer,
    'Swish': Swish,
    'DropConnect': DropConnect
})