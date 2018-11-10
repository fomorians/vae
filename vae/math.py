import tensorflow as tf


def swish(x):
    """
    Self-gating activation function.

    NOTE: TensorFlow already has this function but at the time of writing
          it was not capatible with Eager.
    """
    return x * tf.sigmoid(x)
