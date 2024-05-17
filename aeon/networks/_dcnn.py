import numpy as np
import tensorflow as tf

from aeon.networks.base import BaseDeepNetwork
from aeon.networks.utils import WeightNormalization


def DCNNLayer(inputs, num_filters, layer_num):
    _dilation = 2**layer_num
    _add = tf.keras.layers.Conv1D(num_filters, kernel_size=1)(inputs)
    x = WeightNormalization(
        tf.keras.layers.Conv1D(num_filters, dilation_rate=_dilation, padding="causal")
    )(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = WeightNormalization(
        tf.keras.layers.Conv1D(num_filters, dilation_rate=_dilation, padding="causal")
    )(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    return tf.keras.layers.Add([x, _add])


class TripletLossTimeSeries(tf.keras.losses.Loss):
    def __init__(
        self,
        encoder,
        X,
        anchor_length,
        num_neg_samples,
        negative_penalty,
        fixed_time_dim=False,
        reduction=tf.keras.losses.Reduction.AUTO,
    ):
        super().__init__(reduction)

        self.encoder = encoder
        self.X = X
        self.anchor_length = min(anchor_length, X.shape[1])
        self.num_neg_samples = num_neg_samples  # For Negative Sampling (k)
        self.negative_penalty = negative_penalty
        self.fixed_time_dim = fixed_time_dim

    def call(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true:
            Not to be used but required by Keras.
        y_pred:
            Predicted Embeddings by Encoder.

        Returns
        -------
        float:
            triplet loss between anchor, positive and negative.
        """
        batch_size = tf.shape(y_pred)[0]
        length = self.anchor_length
        fixed_length = tf.shape(self.X)[1]

        # Extract embeddings for positive samples
        positive_embeddings = y_pred[:, :length]

        # Generate negative samples
        negative_indices = np.random.choice(
            self.X.shape[0], self.num_neg_samples * batch_size
        )
        negative_samples = tf.gather(self.X, negative_indices)

        negative_embeddings = []
        for i in range(self.num_neg_samples):
            neg_sample = negative_samples[i * batch_size : (i + 1) * batch_size]
            neg_embeddings = self.encoder(neg_sample)
            neg_embeddings = neg_embeddings[:, :length]
            if self.fixed_time_dim:
                neg_embeddings = tf.pad(
                    neg_embeddings, [[0, 0], [0, fixed_length - length], [0, 0]]
                )
            negative_embeddings.append(neg_embeddings)
        negative_embeddings = tf.stack(negative_embeddings)

        if self.fixed_time_dim:
            positive_embeddings = tf.pad(
                positive_embeddings, [[0, 0], [0, fixed_length - length], [0, 0]]
            )

        # Compute positive loss
        positive_similarity = tf.reduce_sum(y_pred * positive_embeddings, axis=-1)
        positive_loss = -tf.reduce_mean(tf.math.log_sigmoid(positive_similarity))

        # Compute negative loss
        negative_loss = 0
        for neg_embeddings in negative_embeddings:
            negative_similarity = tf.reduce_sum(y_pred * neg_embeddings, axis=-1)
            negative_loss += -tf.reduce_mean(tf.math.log_sigmoid(-negative_similarity))

        negative_loss *= self.negative_penalty / self.num_neg_samples

        loss = positive_loss + negative_loss
        return loss


class DCNNEncoderNetwork(BaseDeepNetwork):
    def __init__(self, latent_dim, kernel_size, num_filters, model_depth, loss_function):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.loss_functions = loss_function
        self.model_depth = model_depth

    def build_network(self, input_shape):
        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer
        for i in range(1, self.model_depth + 1):
            x = DCNNLayer(x, self.num_filters, i)
        x = tf.keras.layers.GlobalMaxPool1D()(x)
        output_layer = tf.keras.layers.Dense(self.latent_dim)(x)

        return tf.keras.Model(inputs=input_layer, outputs = output_layer)
