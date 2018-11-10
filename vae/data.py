import numpy as np
import tensorflow as tf


def prep_images(images):
    # convert uint8 images to float32 and rescale from 0..255 => 0..1
    return images[..., None].astype(np.float32) / 255


def get_dataset(tensors, batch_size=None, shuffle=False, buffer_size=10000):
    # if the batch size is defined return a batched dataset of individual
    # tensor slices, otherwise return a unbatched dataset
    if batch_size is not None:
        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensors(tensors)
    return dataset
