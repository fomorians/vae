import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


def swish(x):
    return x * tf.sigmoid(x)


class Normalizer(tf.keras.layers.Layer):
    def __init__(self, loc, scale):
        super(Normalizer, self).__init__()
        self.loc = tfe.Variable(loc, trainable=False)
        self.scale = tfe.Variable(scale, trainable=False)

    def call(self, inputs):
        return (inputs - self.loc) / self.scale


class Model(tf.keras.Model):
    def __init__(self, inputs_loc, inputs_scale):
        super(Model, self).__init__()

        he_init2 = tf.initializers.variance_scaling(2.0)
        he_init1 = tf.initializers.variance_scaling(1.0)

        self.normalizer = Normalizer(inputs_loc, inputs_scale)
        self.flatten = tf.keras.layers.Flatten()

        # encoder
        self.encoder1 = tf.keras.layers.Dense(
            units=512, activation=swish, kernel_initializer=he_init2)
        self.encoder2 = tf.keras.layers.Dense(
            units=256, activation=swish, kernel_initializer=he_init2)

        self.encoder3_loc = tf.keras.layers.Dense(
            units=128, activation=None, kernel_initializer=he_init1)
        self.encoder3_scale = tf.keras.layers.Dense(
            units=128, activation=None, kernel_initializer=he_init1)

        # decoder
        self.decoder1 = tf.keras.layers.Dense(
            units=256, activation=swish, kernel_initializer=he_init2)
        self.decoder2 = tf.keras.layers.Dense(
            units=512, activation=swish, kernel_initializer=he_init2)

        self.outputs = tf.keras.layers.Dense(
            units=28 * 28, activation=None, kernel_initializer=he_init1)

    def _encode(self, inputs, training=False):
        hidden = self.encoder1(inputs)
        hidden = self.encoder2(hidden)
        z_loc = self.encoder3_loc(hidden)
        z_scale = tf.nn.softplus(
            self.encoder3_scale(hidden) +
            tfp.distributions.softplus_inverse(1.0))
        z_dist = tfp.distributions.MultivariateNormalDiag(
            loc=z_loc, scale_diag=z_scale)
        return z_dist

    def _decode(self, z, training=False):
        hidden = self.decoder1(z)
        hidden = self.decoder2(hidden)
        outputs = self.outputs(hidden)
        logits = tf.reshape(outputs, [-1, 28, 28, 1])
        outputs_dist = tfp.distributions.Independent(
            tfp.distributions.Bernoulli(logits=logits, dtype=tf.float32),
            reinterpreted_batch_ndims=3)
        return outputs_dist

    def call(self, inputs, training=False):
        inputs_norm = self.normalizer(inputs)
        inputs_flat = self.flatten(inputs_norm)
        z_dist = self._encode(inputs_flat, training=training)
        z = z_dist.sample()
        outputs_dist = self._decode(z, training=training)
        outputs = outputs_dist.sample()
        return outputs_dist, outputs, z_dist, z
