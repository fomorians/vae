import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.contrib.eager as tfe

from vae import math


class Normalizer(tf.keras.layers.Layer):
    """
    Standard normalization layer.

    Normalizes inputs to a mean near 0 and a standard deviation near 1.
    """

    def __init__(self, loc, scale):
        super(Normalizer, self).__init__()
        self.loc = tfe.Variable(loc, trainable=False)
        self.scale = tfe.Variable(scale, trainable=False)

    def call(self, inputs):
        return (inputs - self.loc) / self.scale


class Model(tf.keras.Model):
    def __init__(self, inputs_loc, inputs_scale, inputs_shape):
        super(Model, self).__init__()

        self.inputs_shape = inputs_shape

        # define initializers
        # use variance scale of 2.0 for ReLU family
        he_init2 = tf.initializers.variance_scaling(2.0)
        he_init1 = tf.initializers.variance_scaling(1.0)

        # define input layers
        self.normalizer = Normalizer(inputs_loc, inputs_scale)
        self.flatten = tf.keras.layers.Flatten()

        # define encoder layers
        self.encoder1 = tf.keras.layers.Dense(
            units=512, activation=math.swish, kernel_initializer=he_init2)
        self.encoder2 = tf.keras.layers.Dense(
            units=64, activation=math.swish, kernel_initializer=he_init2)

        self.encoder3_loc = tf.keras.layers.Dense(
            units=8, activation=None, kernel_initializer=he_init1)
        self.encoder3_scale = tf.keras.layers.Dense(
            units=8, activation=tf.nn.softplus, kernel_initializer=he_init1)

        # define decoder layers
        self.decoder1 = tf.keras.layers.Dense(
            units=64, activation=math.swish, kernel_initializer=he_init2)
        self.decoder2 = tf.keras.layers.Dense(
            units=512, activation=math.swish, kernel_initializer=he_init2)

        # define output layers
        self.outputs = tf.keras.layers.Dense(
            units=np.prod(inputs_shape),
            activation=None,
            kernel_initializer=he_init1)

    def call(self, inputs, training=False):
        # pre-process inputs
        inputs_norm = self.normalizer(inputs)
        inputs_flat = self.flatten(inputs_norm)

        # run encoder
        encoder1 = self.encoder1(inputs_flat)
        encoder2 = self.encoder2(encoder1)
        z_loc = self.encoder3_loc(encoder2)
        z_scale = self.encoder3_scale(encoder2)
        z_dist = tfp.distributions.MultivariateNormalDiag(
            loc=z_loc, scale_diag=z_scale)

        # sample latent variable
        z = z_dist.sample()

        # run decoder
        decoder1 = self.decoder1(z)
        decoder2 = self.decoder2(decoder1)
        outputs = self.outputs(decoder2)
        logits = tf.reshape(outputs, [-1] + list(self.inputs_shape))

        # use an independent distribution to aggregate the log probability
        # for the loss function:
        outputs_dist = tfp.distributions.Independent(
            tfp.distributions.Bernoulli(logits=logits, dtype=tf.float32),
            reinterpreted_batch_ndims=3)
        return outputs_dist, z_dist, z
