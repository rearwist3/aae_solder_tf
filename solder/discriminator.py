import tensorflow as tf
from layers import dense, dropout
from models import Discriminator as D


class Discriminator(D):
    def __init__(self,
                 is_training=True,
                 is_dropout=True):
        super().__init__(is_training,
                         is_dropout)

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _x = dense(x, 500, activation_='lrelu')
            _x = dense(_x, 500, activation_='lrelu')
            _x = dense(_x, 1, activation_=None)
            return _x
