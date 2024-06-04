"""Encoder Classifier."""

__maintainer__ = ["hadifawaz1999"]

from aeon.networks.base import BaseDeepNetwork
from aeon.utils.networks import tensorflow_addons as types
from aeon.utils.validation._dependencies import _check_soft_dependencies


class EncoderNetwork(BaseDeepNetwork):
    """Establish the network structure for an Encoder.

    Adapted from the implementation used in [1]

    Parameters
    ----------
    kernel_size : array of int, default = [5, 11, 21]
        Specifies the length of the 1D convolution windows.
    n_filters : array of int, default = [128, 256, 512]
        Specifying the number of 1D convolution filters used for each layer,
        the shape of this array should be the same as kernel_size.
    max_pool_size : int, default = 2
        Size of the max pooling windows.
    activation : string, default = sigmoid
        Keras activation function.
    dropout_proba : float, default = 0.2
        specifying the dropout layer probability.
    padding : string, default = "same"
        Specifying the type of padding used for the 1D convolution.
    strides : int, default = 1
        Specifying the sliding rate of the 1D convolution filter.
    fc_units : int, default = 256
        Specifying the number of units in the hiddent fully connected layer used in
        the EncoderNetwork.

    Notes
    -----
    Adapted from source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

    References
    ----------
    .. [1] Serrà et al. Towards a Universal Neural Network Encoder for Time Series
    In proceedings International Conference of the Catalan Association
    for Artificial Intelligence, 120--129 2018.


    """

    _tags = {"python_dependencies": ["tensorflow", "typeguard"]}

    def __init__(
        self,
        kernel_size=None,
        n_filters=None,
        dropout_proba=0.2,
        max_pool_size=2,
        activation="sigmoid",
        padding="same",
        strides=1,
        fc_units=256,
    ):
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.padding = padding
        self.strides = strides
        self.max_pool_size = max_pool_size
        self.activation = activation
        self.dropout_proba = dropout_proba
        self.fc_units = fc_units

        super().__init__()

    def build_network(self, input_shape, **kwargs):
        """
        Construct a network and return its input and output layers.

        Parameters
        ----------
        input_shape : tuple
            The shape of the data fed into the input layer.

        Returns
        -------
        input_layer : a keras layer
        output_layer : a keras layer
        """
        import tensorflow as tf

        tf.keras.config.enable_unsafe_deserialization()

        self._kernel_size = (
            [5, 11, 21] if self.kernel_size is None else self.kernel_size
        )
        self._n_filters = [128, 256, 512] if self.n_filters is None else self.n_filters

        input_layer = tf.keras.layers.Input(input_shape)

        x = input_layer

        for i in range(len(self._kernel_size)):
            conv = tf.keras.layers.Conv1D(
                filters=self._n_filters[i],
                kernel_size=self._kernel_size[i],
                padding=self.padding,
                strides=self.strides,
            )(x)

            conv = InstanceNormalization()(conv)
            conv = tf.keras.layers.PReLU(shared_axes=[1])(conv)
            conv = tf.keras.layers.Dropout(self.dropout_proba)(conv)

            if i < len(self._kernel_size) - 1:
                conv = tf.keras.layers.MaxPool1D(pool_size=self.max_pool_size)(conv)

            x = conv

        # split attention

        split_index = self._n_filters[-1] // 2

        attention_multiplier_1 = tf.keras.layers.Softmax()(
            tf.keras.layers.Lambda(lambda x: x[:, :, :split_index])(conv)
        )
        attention_multiplier_2 = tf.keras.layers.Lambda(
            lambda x: x[:, :, split_index:]
        )(conv)

        # attention mechanism

        attention = tf.keras.layers.Multiply()(
            [attention_multiplier_1, attention_multiplier_2]
        )

        # add fully connected hidden layer

        hidden_fc_layer = tf.keras.layers.Dense(
            units=self.fc_units, activation=self.activation
        )(attention)
        hidden_fc_layer = InstanceNormalization()(hidden_fc_layer)

        # output layer before classification layer

        flatten_layer = tf.keras.layers.Flatten()(hidden_fc_layer)

        return input_layer, flatten_layer


if _check_soft_dependencies(["tensorflow"], severity="none"):
    import logging

    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="Addons")
    class GroupNormalization(tf.keras.layers.Layer):
        """Group normalization layer.

        Source: "Group Normalization" (Yuxin Wu & Kaiming He, 2018)
        https://arxiv.org/abs/1803.08494

        Group Normalization divides the channels into groups and computes
        within each group the mean and variance for normalization.
        Empirically, its accuracy is more stable than batch norm in a wide
        range of small batch sizes, if learning rate is adjusted linearly
        with batch sizes.

        Relation to Layer Normalization:
        If the number of groups is set to 1, then this operation becomes identical
        to Layer Normalization.

        Relation to Instance Normalization:
        If the number of groups is set to the
        input dimension (number of groups is equal
        to number of channels), then this operation becomes
        identical to Instance Normalization.

        Parameters
        ----------
            groups: Integer, the number of groups for Group Normalization.
                Can be in the range [1, N] where N is the input dimension.
                The input dimension must be divisible by the number of groups.
                Defaults to 32.
            axis: Integer, the axis that should be normalized.
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor.
                If False, `beta` is ignored.
            scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.

        Notes
        -----
        This code was taken from the soon to be deprecated project
        tensorflow_addons:
        https://github.com/tensorflow/addons/tree/v0.20.0
        """

        def __init__(
            self,
            groups: int = 32,
            axis: int = -1,
            epsilon: float = 1e-3,
            center: bool = True,
            scale: bool = True,
            beta_initializer: types.Initializer = "zeros",
            gamma_initializer: types.Initializer = "ones",
            beta_regularizer: types.Regularizer = None,
            gamma_regularizer: types.Regularizer = None,
            beta_constraint: types.Constraint = None,
            gamma_constraint: types.Constraint = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.axis = axis
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = tf.keras.initializers.get(beta_initializer)
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
            self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = tf.keras.constraints.get(beta_constraint)
            self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
            self._check_axis()

        def build(self, input_shape):

            self._check_if_input_shape_is_none(input_shape)
            self._set_number_of_groups_for_instance_norm(input_shape)
            self._check_size_of_dimensions(input_shape)
            self._create_input_spec(input_shape)

            self._add_gamma_weight(input_shape)
            self._add_beta_weight(input_shape)
            self.built = True
            super().build(input_shape)

        def call(self, inputs):

            input_shape = tf.keras.backend.int_shape(inputs)
            tensor_input_shape = tf.shape(inputs)

            reshaped_inputs, group_shape = self._reshape_into_groups(
                inputs, input_shape, tensor_input_shape
            )

            normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

            is_instance_norm = (input_shape[self.axis] // self.groups) == 1
            if not is_instance_norm:
                outputs = tf.reshape(normalized_inputs, tensor_input_shape)
            else:
                outputs = normalized_inputs

            return outputs

        def get_config(self):
            config = {
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(
                    self.beta_initializer
                ),
                "gamma_initializer": tf.keras.initializers.serialize(
                    self.gamma_initializer
                ),
                "beta_regularizer": tf.keras.regularizers.serialize(
                    self.beta_regularizer
                ),
                "gamma_regularizer": tf.keras.regularizers.serialize(
                    self.gamma_regularizer
                ),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(
                    self.gamma_constraint
                ),
            }
            base_config = super().get_config()
            return {**base_config, **config}

        def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

            group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
            is_instance_norm = (input_shape[self.axis] // self.groups) == 1
            if not is_instance_norm:
                group_shape[self.axis] = input_shape[self.axis] // self.groups
                group_shape.insert(self.axis, self.groups)
                group_shape = tf.stack(group_shape)
                reshaped_inputs = tf.reshape(inputs, group_shape)
                return reshaped_inputs, group_shape
            else:
                return inputs, group_shape

        def _apply_normalization(self, reshaped_inputs, input_shape):

            group_shape = tf.keras.backend.int_shape(reshaped_inputs)
            group_reduction_axes = list(range(1, len(group_shape)))
            is_instance_norm = (input_shape[self.axis] // self.groups) == 1
            if not is_instance_norm:
                axis = -2 if self.axis == -1 else self.axis - 1
            else:
                axis = -1 if self.axis == -1 else self.axis - 1
            group_reduction_axes.pop(axis)

            mean, variance = tf.nn.moments(
                reshaped_inputs, group_reduction_axes, keepdims=True
            )

            gamma, beta = self._get_reshaped_weights(input_shape)
            normalized_inputs = tf.nn.batch_normalization(
                reshaped_inputs,
                mean=mean,
                variance=variance,
                scale=gamma,
                offset=beta,
                variance_epsilon=self.epsilon,
            )
            return normalized_inputs

        def _get_reshaped_weights(self, input_shape):
            broadcast_shape = self._create_broadcast_shape(input_shape)
            gamma = None
            beta = None
            if self.scale:
                gamma = tf.reshape(self.gamma, broadcast_shape)

            if self.center:
                beta = tf.reshape(self.beta, broadcast_shape)
            return gamma, beta

        def _check_if_input_shape_is_none(self, input_shape):
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError(
                    "Axis " + str(self.axis) + " of "
                    "input tensor should have a defined dimension "
                    "but the layer received an input with shape "
                    + str(input_shape)
                    + "."
                )

        def _set_number_of_groups_for_instance_norm(self, input_shape):
            dim = input_shape[self.axis]

            if self.groups == -1:
                self.groups = dim

        def _check_size_of_dimensions(self, input_shape):

            dim = input_shape[self.axis]
            if dim < self.groups:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") cannot be "
                    "more than the number of channels (" + str(dim) + ")."
                )

            if dim % self.groups != 0:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") must be a "
                    "multiple of the number of channels (" + str(dim) + ")."
                )

        def _check_axis(self):

            if self.axis == 0:
                raise ValueError(
                    "You are trying to normalize your batch axis. Do you want to "
                    "use tf.layer.batch_normalization instead"
                )

        def _create_input_spec(self, input_shape):

            dim = input_shape[self.axis]
            self.input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )

        def _add_gamma_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(
                    shape=shape,
                    name="gamma",
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    constraint=self.gamma_constraint,
                )
            else:
                self.gamma = None

        def _add_beta_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.center:
                self.beta = self.add_weight(
                    shape=shape,
                    name="beta",
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    constraint=self.beta_constraint,
                )
            else:
                self.beta = None

        def _create_broadcast_shape(self, input_shape):
            broadcast_shape = [1] * len(input_shape)
            is_instance_norm = (input_shape[self.axis] // self.groups) == 1
            if not is_instance_norm:
                broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
                broadcast_shape.insert(self.axis, self.groups)
            else:
                broadcast_shape[self.axis] = self.groups
            return broadcast_shape

    @tf.keras.utils.register_keras_serializable(package="Addons")
    class InstanceNormalization(GroupNormalization):
        """Instance normalization layer.

        Instance Normalization is an specific case of ```GroupNormalization```since
        it normalizes all features of one channel. The Groupsize is equal to the
        channel size. Empirically, its accuracy is more stable than batch norm in a
        wide range of small batch sizes, if learning rate is adjusted linearly
        with batch sizes.

        Parameters
        ----------
            axis: Integer, the axis that should be normalized.
            epsilon: Small float added to variance to avoid dividing by zero.
            center: If True, add offset of `beta` to normalized tensor.
                If False, `beta` is ignored.
            scale: If True, multiply by `gamma`.
                If False, `gamma` is not used.
            beta_initializer: Initializer for the beta weight.
            gamma_initializer: Initializer for the gamma weight.
            beta_regularizer: Optional regularizer for the beta weight.
            gamma_regularizer: Optional regularizer for the gamma weight.
            beta_constraint: Optional constraint for the beta weight.
            gamma_constraint: Optional constraint for the gamma weight.

        References
        ----------
            - [Instance Normalization: The Missing Ingredient for Fast Stylization]
            (https://arxiv.org/abs/1607.08022)
        """

        def __init__(self, **kwargs):
            if "groups" in kwargs:
                logging.warning("The given value for groups will be overwritten.")

            kwargs["groups"] = -1
            super().__init__(**kwargs)
