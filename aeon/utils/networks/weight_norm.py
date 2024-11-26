"""Weight Normalization Layer."""

from aeon.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    class _WeightNormalization(tf.keras.layers.Wrapper):
        """Apply weight normalization to a Keras layer."""

        def __init__(self, layer, **kwargs):
            """Initialize the WeightNormalization wrapper.

            Args:
                layer: tf.keras.layers.Layer
                    The Keras layer to apply weight normalization to.
            """
            if not isinstance(layer, tf.keras.layers.Layer):
                raise ValueError("The `layer` argument should be a Keras layer.")

            super().__init__(layer, **kwargs)

        def build(self, input_shape):
            """Build the weight normalization layer.

            This method initializes weights `v` and `g` for weight normalization.
            """
            if not self.layer.built:
                self.layer.build(input_shape)

            self.w = self.layer.kernel
            self.v = self.add_weight(
                shape=self.w.shape,
                initializer="random_normal",
                trainable=True,
                name="v",
            )
            self.g = self.add_weight(
                shape=(self.w.shape[-1],), initializer="ones", trainable=True, name="g"
            )
            super().build(input_shape)

        def call(self, inputs):
            """Apply the normalized weights to the inputs."""
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=0, keepdims=True))
            normalized_kernel = self.g * self.v / norm
            output = tf.nn.conv1d(
                inputs, filters=normalized_kernel, stride=1, padding="SAME"
            )
            return output

        def get_config(self):
            """Return the config of the layer for serialization."""
            base_config = super().get_config()
            return {**base_config, "layer": tf.keras.layers.serialize(self.layer)}

        @classmethod
        def from_config(cls, config):
            """Recreate the layer from its config."""
            layer = tf.keras.layers.deserialize(config.pop("layer"))
            return cls(layer, **config)
