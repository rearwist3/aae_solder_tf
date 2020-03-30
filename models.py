import tensorflow as tf
from tensorflow.python.keras.regularizers import l2


class AutoEncoder:
    def __init__(self, input_shape,
                 latent_dim,
                 last_activation='tanh',
                 color_mode='rgb',
                 normalization='batch',
                 upsampling='deconv',
                 is_dropout=False,
                 is_training=True):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.last_activation = last_activation
        self.name = 'model/generator'
        assert color_mode in ['grayscale', 'gray', 'rgb']
        self.channel = 1 if color_mode in ['grayscale', 'gray'] else 3
        self.normalization = normalization
        self.upsampling = upsampling
        self.is_dropout = is_dropout
        self.is_training = is_training

        self.conv_block_params = {'kernel_initializer': 'he_normal',
                                  'kernel_regularizer': l2(1.e-4),
                                  'activation_': 'lrelu',
                                  'normalization': self.normalization,
                                  'dropout_rate': 0.5 if is_training and is_dropout else 0.,
                                  'is_training': self.is_training}

        self.last_conv_block_params = {'kernel_initializer': 'he_normal',
                                       'kernel_regularizer': l2(1.e-4),
                                       'activation_': self.last_activation,
                                       'normalization': None,
                                       'dropout_rate': 0.,
                                       'is_training': self.is_training}

    def __call__(self, x, reuse=False):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]

    @property
    def encoder_vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name and 'Encoder' in var.name]


class Discriminator:
    def __init__(self,
                 is_training=True,
                 is_dropout=False):
        self.name = 'model/discriminator'
        self.is_training = is_training
        self.is_dropout = is_dropout

    def __call__(self, x, reuse=True):
        raise NotImplementedError

    @property
    def vars(self):
        return [var for var in tf.trainable_variables() if self.name in var.name]
