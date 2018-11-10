import os
import attr
import random
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp

from tqdm import trange

from vae import losses
from vae.data import prep_images, get_dataset
from vae.model import Model


@attr.s
class Params:
    """
    Container for hyperparameters.
    """
    learning_rate = attr.ib(default=1e-3)
    epochs = attr.ib(default=10)
    batch_size = attr.ib(default=1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--seed', default=67, type=int)
    args = parser.parse_args()
    print('args:', args)

    # create a job directory if it doesn't already exist
    if not os.path.exists(args.job_dir):
        os.makedirs(args.job_dir)

    # enable eager execution
    tf.enable_eager_execution()

    # set random seeds for consistent execution
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # define hyperparameters
    params = Params()
    print('params:', params)

    # load MNIST dataset
    (images_train, _), (images_test, _) = tf.keras.datasets.mnist.load_data()

    # prepare the images by casting and rescaling
    images_train = prep_images(images_train)
    images_test = prep_images(images_test)

    # compute statistics from the training set
    images_loc = images_train.mean()
    images_scale = images_train.std()

    # define datasets for sampling batches
    dataset_train = get_dataset(
        images_train, batch_size=params.batch_size, shuffle=True)
    dataset_test = get_dataset(images_test, batch_size=params.batch_size)

    # model / optimization
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    model = Model(
        inputs_loc=images_loc,
        inputs_scale=images_scale,
        inputs_shape=[28, 28, 1])
    latent_prior = tfp.distributions.MultivariateNormalDiag(
        loc=tf.zeros(shape=[8], dtype=tf.float32),
        scale_identity_multiplier=1.0)

    # checkpoints
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, model=model, global_step=global_step)
    checkpoint_path = tf.train.latest_checkpoint(args.job_dir)
    if checkpoint_path is not None:
        checkpoint.restore(checkpoint_path).assert_consumed()

    # summaries
    summary_writer = tf.contrib.summary.create_file_writer(
        args.job_dir, max_queue=1, flush_millis=1000)
    summary_writer.set_as_default()

    with trange(params.epochs) as pbar:
        for epoch in pbar:
            loss_train = tfe.metrics.Mean(name='loss/train')
            for images in dataset_train:
                with tf.GradientTape() as tape:
                    outputs_dist, z_dist, z = model(images, training=True)
                    loss = losses.variational(outputs_dist, z_dist, images,
                                              latent_prior)
                    loss_train(loss)

                grads = tape.gradient(loss, model.trainable_variables)
                grads_and_vars = zip(grads, model.trainable_variables)
                optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)

            with tf.contrib.summary.always_record_summaries():
                loss_train.result()

                tf.contrib.summary.scalar(
                    name='grad_norm', tensor=tf.global_norm(grads))

                tf.contrib.summary.image(
                    name='image/train',
                    tensor=images,
                    max_images=2,
                    step=global_step)
                tf.contrib.summary.image(
                    name='outputs/train',
                    tensor=outputs_dist.mean(),
                    max_images=2,
                    step=global_step)

            loss_test = tfe.metrics.Mean(name='loss/eval')
            for images in dataset_test:
                outputs_dist, z_dist, z = model(images)
                loss = losses.variational(outputs_dist, z_dist, images,
                                          latent_prior)
                loss_test(loss)

            with tf.contrib.summary.always_record_summaries():
                loss_test.result()

                tf.contrib.summary.image(
                    name='image/eval',
                    tensor=images,
                    max_images=2,
                    step=global_step)
                tf.contrib.summary.image(
                    name='outputs/eval',
                    tensor=outputs_dist.mean(),
                    max_images=2,
                    step=global_step)

            pbar.set_description('loss (train): {}, loss (eval): {}'.format(
                loss_train.result().numpy(),
                loss_test.result().numpy()))

            checkpoint_prefix = os.path.join(args.job_dir, 'ckpt')
            checkpoint.save(checkpoint_prefix)


if __name__ == '__main__':
    main()
