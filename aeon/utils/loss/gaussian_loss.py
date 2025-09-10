"""Gaussian Loss function."""

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import numpy as np
    from tensorflow.keras.utils import register_keras_serializable

    @register_keras_serializable(package="aeon")
    def gaussian_loss(y_true, y_pred):
        """
        Gaussian negative log-likelihood loss for concatenated output format.

        Parameters
        ----------
        y_true: Ground truth values of shape (batch_size, 2, n_channels)
        y_pred: Predicted parameters of shape (batch_size, 2, n_channels)
                where y_pred[:, 0, :] = mu and y_pred[:, 1, :] = sigma
        """
        import tensorflow as tf

        mu = y_pred[:, 0, :]
        sigma = y_pred[:, 1, :]

        sigma = tf.maximum(sigma, 1e-6)
        mu_true = y_true[:, 0, :]
        return tf.reduce_mean(
            tf.math.log(tf.math.sqrt(2 * tf.constant(np.pi)))
            + tf.math.log(sigma)
            + tf.truediv(tf.square(mu_true - mu), 2 * tf.square(sigma))
        )
