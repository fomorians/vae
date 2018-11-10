import tensorflow as tf
import tensorflow_probability as tfp


def variational(outputs_dist, z_dist, targets, latent_prior):
    # compute log probability of targets
    log_prob = outputs_dist.log_prob(targets)

    # compute kl divergence between the latent distribution and prior
    kl = tfp.distributions.kl_divergence(z_dist, latent_prior)

    # compute ELBO and loss
    elbo = tf.reduce_mean(log_prob - kl)
    loss = -elbo

    # ensure no invalid numbers
    loss = tf.check_numerics(loss, 'loss')
    return loss
