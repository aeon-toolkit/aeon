"""Residual Network (ResNetNetwork)."""

__maintainer__ = ["hadifawaz1999"]


from aeon.networks.base import BaseDeepLearningNetwork
from aeon.typing import RESNET_TYPE
from aeon.utils.validation._dependencies import _check_soft_dependencies

_UNSET = object()

if _check_soft_dependencies(["tensorflow"], severity="none"):
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="aeon")
    class ConstantMultiply(tf.keras.layers.Layer):
        def __init__(self, w, **kwargs):
            super().__init__(**kwargs)
            self.w = w

        def call(self, inputs):
            return inputs * self.w

        def get_config(self):
            config = super().get_config()
            config.update({"w": self.w})
            return config


class L:
    """Layer types for ResNet."""

    CONV = "conv"
    BN = "bn"
    SE = "se"
    RES = "res"
    ACT = "act"
    DPO = "dpo"
    POOL = "pool"


class ResNetNetwork(BaseDeepLearningNetwork):
    """
    Establish the network structure for a ResNet.

    Adapted from the implementations used in [1]_.

    Parameters
    ----------
    n_residual_blocks : int, default = 3
        The number of residual blocks of ResNet's model.
    n_conv_per_residual_block : int or list of int, default = 3
        The number of convolution blocks in each residual block.
        If not a list, all the blocks will share the same number of convolution layers.
    n_filters : int or list of int or list of list of int,
        default = [[64, 64, 64], [128, 128, 128], [128, 128, 128]]
        If int, it sets the same number of convolution filters for all the convolution
        layers.
        If a list of int, each value specifies the number of filters used
        per convolutional layers (identical across each residual block).
        If list of list of int, each value specifies the number of filters used
        in convolutional layers for every residual block, one by one.
    kernel_size : int or list of int or list of list of int, default = [8, 5, 3]
        If int, it sets the same kernel size for all the convolution layers.
        If a list of int, each value specifies the kernel size used per convolutional
        layers (identical across each residual block).
        If list of list of int, each value specifies the kernel size used in
        convolutional layers for every residual block, one by one.
    strides : int or list of int or list of list of int, default = 1
        If int, it sets the same stride for all the convolution layers.
        If a list of int, each value specifies the stride used per convolutional layers
        (identical across each residual block).
        If list of list of int, each value specifies the stride used in convolutional
        layers for every residual block, one by one.
    dilation_rate : int or list of int or list of list of int, default = 1
        If int, it sets the same dilation rate for all the convolution layers.
        If a list of int, each value specifies the dilation rate used per convolutional
        layers (identical across each residual block).
        If list of list of int, each value specifies the dilation rate used in
        convolutional layers for every residual block, one by one.
    padding : str or list of str or list of list of str, default = 'padding'
        If str, it sets the same padding for all the convolution layers.
        If a list of str, each value specifies the padding used per convolutional
        layers (identical across each residual block).
        If list of list of str, each value specifies the padding used in convolutional
        layers for every residual block, one by one.
    activation : str or list of str or list of list of str, default = 'relu'
        If str, it sets the same activation function for all the convolution layers.
        If a list of str, each value specifies the activation function used per
        convolutional layers (identical across each residual block).
        If list of list of str, each value specifies the activation function used in
        convolutional layers for every residual block, one by one.
    use_batch_norm : bool or list of bool or list of list of bool, default = True
        If bool, it sets the same use_batch_norm for all the convolution layers.
        If a list of bool, each value specifies the use_batch_norm used per
        convolutional layers (identical across each residual block).
        If list of list of bool, each value specifies the use_batch_norm used in
        convolutional layers for every residual block, one by one.
    dropout : float or list of float or list of list of float, default = 0
        If float, it sets the same dropout rate for all the convolution layers.
        If a list of float, each value specifies the dropout rate used per convolutional
        layers (identical across each residual block).
        If list of list of float, each value specifies the dropout rate used in
        convolutional layers for every residual block, one by one.
    use_residual : float or list of float, default = True
        If float, it sets the same use_residual for all the residual blocks.
        If a list of float, each value specifies the use_residual per residual block.
        A value of 0 means no residual connection, a value of 1 means a standard
        residual connection, other values indicate a weighted residual connection.
    max_pooling : bool or list of bool, default = False
        Whether to use max pooling after the residual blocks.
        If a list of bool, each value specifies the use of max pooling per residual
        block.
    use_bias : bool or list of bool or list of list of bool, default = True
        if bool, it sets the same use_bias for all the convolution layers.
        If a list of bool, each value specifies if bias is used, per convolutional
        layers (identical across each residual block).
        If list of list of bool, each value specifies the use_bias used in convolutional
        layers for every residual block, one by one.
    cardinality : int or list of int or list of list of int, default = 1
        The number of groups for grouped convolution. If cardinality > 1, the network
        is considered a ResNeXt network.
        if int, it sets the same cardinality for all the convolution layers.
        If a list of int, each value specifies the cardinality used per convolutional
        layers (identical across each residual block).
        If list of list of int, each value specifies the cardinality used in
        convolutional layers for every residual block, one by one.
    se_ratio : int or list of int, default = 0
        Whether to use Squeeze-and-Excitation (SE) blocks after the residual blocks.
        If a list of int, each value specifies the use of SE block per residual block.
        A value of 0 means no SE block, other values indicate
        the reduction ratio for the SE block (generally 16).
    resnet_type : RESNET_TYPE, default = RESNET_TYPE.V1
        The type of ResNet to use. Options are:
        - RESNET_TYPE.V1: The original ResNet architecture.
        - RESNET_TYPE.V2: The pre-activation ResNet architecture.
    transpose : bool, default = False
        Whether or not to use transposed convolution layers instead of
        convolution layers. When transpose is True, MaxPooling layers
        are replaced with UpSampling layers.
    preset : ResNetNetwork.Presets, default = None
        A preset configuration for the ResNetNetwork.
        Provided parameters will override the preset configuration.


    Notes
    -----
    Adapted from the implementation source code
    https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/resnet.py

    References
    ----------
    .. [1] H. Fawaz, G. B. Lanckriet, F. Petitjean, and L. Idoumghar,

    Network originally defined in:

    @inproceedings{wang2017time, title={Time series classification from
    scratch with deep neural networks: A strong baseline}, author={Wang,
    Zhiguang and Yan, Weizhong and Oates, Tim}, booktitle={2017
    International joint conference on neural networks (IJCNN)}, pages={
    1578--1585}, year={2017}, organization={IEEE} }

    """

    _config = {
        **BaseDeepLearningNetwork._config,
        "structure": "encoder",
    }

    _default = {
        "n_residual_blocks": 3,
        "n_conv_per_residual_block": 3,
        "n_filters": [[64, 64, 64], [128, 128, 128], [128, 128, 128]],
        "kernel_size": [8, 5, 3],
        "strides": 1,
        "dilation_rate": 1,
        "padding": "same",
        "activation": "relu",
        "use_batch_norm": True,
        "dropout": 0,
        "use_residual": True,
        "max_pooling": False,
        "use_bias": True,
        "cardinality": 1,
        "se_ratio": 0,
        "resnet_type": RESNET_TYPE.V1,
        "transpose": False,
    }

    def __init__(
        self,
        n_residual_blocks=_UNSET,
        n_conv_per_residual_block=_UNSET,
        n_filters=_UNSET,
        kernel_size=_UNSET,
        strides=_UNSET,
        dilation_rate=_UNSET,
        padding=_UNSET,
        activation=_UNSET,
        use_batch_norm=_UNSET,
        dropout=_UNSET,
        use_residual=_UNSET,
        max_pooling=_UNSET,
        use_bias=_UNSET,
        resnet_type=_UNSET,
        cardinality=_UNSET,
        se_ratio=_UNSET,
        transpose=_UNSET,
        preset=None,
    ):
        super().__init__()

        # start from default parameters
        param = self._default.copy()

        # overload with preset if provided
        if preset is not None:
            preset_params = self.get_preset(preset)
            param.update(preset_params)

        # overload with provided parameters
        for key in param.keys():
            if locals()[key] is not _UNSET:
                param[key] = locals()[key]

        self.n_residual_blocks = param["n_residual_blocks"]
        self.n_conv_per_residual_block = param["n_conv_per_residual_block"]
        self.n_filters = param["n_filters"]
        self.kernel_size = param["kernel_size"]
        self.strides = param["strides"]
        self.dilation_rate = param["dilation_rate"]
        self.padding = param["padding"]
        self.activation = param["activation"]
        self.use_batch_norm = param["use_batch_norm"]
        self.dropout = param["dropout"]
        self.use_residual = param["use_residual"]
        self.max_pooling = param["max_pooling"]
        self.use_bias = param["use_bias"]
        self.resnet_type = param["resnet_type"]
        self.cardinality = param["cardinality"]
        self.se_ratio = param["se_ratio"]
        self.transpose = param["transpose"]

    def _check_params(self):
        b, c = self.n_residual_blocks, self.n_conv_per_residual_block
        block_name = "residual block"
        depth_name = "conv"
        nb_res = "number of residual blocks"
        default_n_filters = [[64, 64, 64], [128, 128, 128], [128, 128, 128]]

        arg = {
            "block_name": block_name,
            "block_depth_name": depth_name,
        }

        self._n_filters = BaseDeepLearningNetwork._check_block_param(
            b, c, self.n_filters, "filters", default_n_filters, **arg
        )
        self._kernel_size = BaseDeepLearningNetwork._check_block_param(
            b, c, self.kernel_size, "kernels", [8, 5, 3], **arg
        )
        self._strides = BaseDeepLearningNetwork._check_block_param(
            b, c, self.strides, "strides", 1, **arg
        )
        self._dilation_rate = BaseDeepLearningNetwork._check_block_param(
            b, c, self.dilation_rate, "dilations", 1, **arg
        )
        self._padding = BaseDeepLearningNetwork._check_block_param(
            b, c, self.padding, "paddings", "same", **arg
        )
        self._activation = BaseDeepLearningNetwork._check_block_param(
            b, c, self.activation, "activations", default=None, allow_none=True, **arg
        )
        self._use_batch_norm = BaseDeepLearningNetwork._check_block_param(
            b, c, self.use_batch_norm, "batch_norm", True, **arg
        )
        self._dropout = BaseDeepLearningNetwork._check_block_param(
            b, c, self.dropout, "dropout", 0, **arg
        )
        self._use_bias = BaseDeepLearningNetwork._check_block_param(
            b, c, self.use_bias, "biases", True, **arg
        )
        self._use_residual = BaseDeepLearningNetwork._check_layer_param(
            b, self.use_residual, "use_residual", default=True, same_as=nb_res
        )
        self._max_pooling = BaseDeepLearningNetwork._check_layer_param(
            b, self.max_pooling, "max_pooling", default=False, same_as=nb_res
        )
        self._n_conv_per_residual_block = BaseDeepLearningNetwork._check_layer_param(
            b, c, "number of convolution layers", default=3, same_as=nb_res
        )
        self._cardinality = BaseDeepLearningNetwork._check_block_param(
            b, c, self.cardinality, "cardinality", 1, **arg
        )
        self._se_ratio = BaseDeepLearningNetwork._check_layer_param(
            b, self.se_ratio, "se_ratio", default=False, same_as=nb_res
        )
        self._resnet_type = RESNET_TYPE._check_params(self.resnet_type)

    @staticmethod
    def _get_layer_order(resnet_type: RESNET_TYPE):
        match resnet_type:
            case RESNET_TYPE.V1:
                return [L.CONV, L.BN, L.SE, L.RES, L.ACT, L.DPO, L.POOL]
            case RESNET_TYPE.V2:
                return [L.CONV, L.POOL, L.SE, L.RES, L.BN, L.ACT, L.DPO]
            case _:
                raise ValueError(f"Unknown resnet_type: {resnet_type}")

    def _build_resnet_block(self, order, x, b, skip, pre_skip=None):
        for c in range(self._n_conv_per_residual_block[b]):
            for i, l in enumerate(order):
                if l == L.CONV:
                    if not self.transpose:
                        x = tf.keras.layers.Conv1D(
                            filters=self._n_filters[b][c],
                            kernel_size=self._kernel_size[b][c],
                            strides=self._strides[b][c],
                            padding=self._padding[b][c],
                            dilation_rate=self._dilation_rate[b][c],
                            use_bias=self._use_bias[b][c],
                            groups=self._cardinality[b][c],
                        )(x)
                    else:
                        x = tf.keras.layers.Conv1DTranspose(
                            filters=self._n_filters[b][c],
                            kernel_size=self._kernel_size[b][c],
                            strides=self._strides[b][c],
                            padding=self._padding[b][c],
                            dilation_rate=self._dilation_rate[b][c],
                            use_bias=self._use_bias[b][c],
                        )(x)

                if l == L.POOL and c == 0 and self._max_pooling[b]:
                    if not self.transpose:
                        x = tf.keras.layers.MaxPooling1D(
                            pool_size=2, strides=2, padding="same"
                        )(x)
                    else:
                        x = tf.keras.layers.UpSampling1D(size=2)(x)

                if l == L.BN and self._use_batch_norm[b][c]:
                    x = tf.keras.layers.BatchNormalization()(x)

                if l == L.SE and c == self._n_conv_per_residual_block[b] - 1:
                    if self._use_residual[b] != 0:
                        x = self._build_squeeze_excitation_block(x, b)

                if l == L.RES and c == self._n_conv_per_residual_block[b] - 1:
                    # resnet v2 uses pre-activation for the projection shortcut
                    if self._is_identity(x, skip):
                        x = self._identity_shortcut(x, skip, w=self._use_residual[b])
                    else:
                        if self._resnet_type == RESNET_TYPE.V1:
                            x = self._projection_shortcut(x, skip, b, c)
                        elif self._resnet_type == RESNET_TYPE.V2:
                            x = self._projection_shortcut(x, pre_skip, b, c)

                    pre_skip = None
                    if self._resnet_type == RESNET_TYPE.V1:
                        skip = None
                    elif self._resnet_type == RESNET_TYPE.V2:
                        skip = x

                if l == L.ACT:
                    if self._activation[b][c] is not None:
                        x = tf.keras.layers.Activation(self._activation[b][c])(x)

                if l == L.DPO and self._dropout[b][c] > 0:
                    x = tf.keras.layers.Dropout(self._dropout[b][c])(x)

                if i == len(order) - 1:
                    if skip is None:
                        skip = x
                    if pre_skip is None:
                        pre_skip = x

        return x, skip, pre_skip

    @staticmethod
    def _is_identity(x, skip):
        return (skip.shape[-1] == x.shape[-1]) and (skip.shape[1] == x.shape[1])

    def _identity_shortcut(self, x, skip, w=1):
        import tensorflow as tf

        if w == 0:
            return x

        if w != 1:
            skip = ConstantMultiply(w)(skip)
        return tf.keras.layers.Add()([x, skip])

    @staticmethod
    def _compute_projection_strides(x, skip):
        strides, s_remainder = divmod(skip.shape[1], x.shape[1])
        upsampling, u_remainder = divmod(x.shape[1], skip.shape[1])
        if s_remainder != 0 and u_remainder != 0:
            raise ValueError(
                f"Cannot compute strides for skip connection between layers: "
                f"skip.shape[1] = {skip.shape[1]}, x.shape[1] = {x.shape[1]}"
            )
        if s_remainder != 0:
            strides = 1
        if u_remainder != 0:
            upsampling = 1
        return strides, upsampling

    def _projection_shortcut(self, x, skip, b, c):
        import tensorflow as tf

        w = self._use_residual[b]
        pad = self._padding[b][c]
        bias = self._use_bias[b][c]
        use_bn = self._use_batch_norm[b][c]

        if w == 0:
            return x

        strides, upsampling = self._compute_projection_strides(x, skip)
        if upsampling > 1:
            skip = tf.keras.layers.UpSampling1D(size=upsampling)(skip)
            if skip.shape[-1] != x.shape[-1]:
                skip = tf.keras.layers.Conv1D(x.shape[-1], 1, 1, pad, use_bias=bias)(
                    skip
                )
        else:
            skip = tf.keras.layers.Conv1D(x.shape[-1], 1, strides, pad, use_bias=bias)(
                skip
            )

        if self._resnet_type == RESNET_TYPE.V1 and use_bn:
            skip = tf.keras.layers.BatchNormalization()(skip)

        if w != 1:
            skip = ConstantMultiply(w)(skip)

        return tf.keras.layers.Add()([x, skip])

    def _build_squeeze_excitation_block(self, x, b):
        import tensorflow as tf

        if not self._se_ratio[b]:
            return x

        channels = x.shape[-1]
        hidden = max(channels // self._se_ratio[b], 1)

        se = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(x)
        se = tf.keras.layers.Conv1D(hidden, 1, activation="relu")(se)
        se = tf.keras.layers.Conv1D(channels, 1, activation="sigmoid")(se)

        return tf.keras.layers.Multiply()([x, se])

    def build_base_graph(self, x):
        self._check_params()
        order = self._get_layer_order(self._resnet_type)
        skip, pre_skip = x, x
        for b in range(self.n_residual_blocks):
            x, skip, pre_skip = self._build_resnet_block(order, x, b, skip, pre_skip)
        return x

    RESNET_18 = "resnet-18"
    RESNET_34 = "resnet-34"
    RESNET_50 = "resnet-50"
    RESNET_101 = "resnet-101"
    RESNET_152 = "resnet-152"

    RESNET_V15_50 = "resnet_v1.5-50"
    RESNET_V15_101 = "resnet_v1.5-101"
    RESNET_V15_152 = "resnet_v1.5-152"

    RESNET_V2_18 = "resnet_v2-18"
    RESNET_V2_34 = "resnet_v2-34"
    RESNET_V2_50 = "resnet_v2-50"
    RESNET_V2_101 = "resnet_v2-101"
    RESNET_V2_152 = "resnet_v2-152"

    W_RESNET_16_8 = "w_resnet-16-8"
    W_RESNET_16_10 = "w_resnet-16-10"
    W_RESNET_22_8 = "w_resnet-22-8"
    W_RESNET_22_10 = "w_resnet-22-10"
    W_RESNET_28_10 = "w_resnet-28-10"
    W_RESNET_28_12 = "w_resnet-28-10"
    W_RESNET_40_1 = "w_resnet-40-1"
    W_RESNET_40_2 = "w_resnet-40-2"
    W_RESNET_40_4 = "w_resnet-40-4"
    W_RESNET_40_8 = "w_resnet-40-8"

    RESNEXT_29_8_64D = "resnext_29_8x64d"
    RESNEXT_29_16_64D = "resnext_29_16x64d"
    RESNEXT_29_32_4D = "resnext_29_32x64d"
    RESNEXT_50_2_40D = "resnext_50_2x40d"
    RESNEXT_50_4_24D = "resnext_50_4x24d"
    RESNEXT_50_8_14D = "resnext_50_8x14d"
    RESNEXT_50_32_4D = "resnext_50_32x4d"
    RESNEXT_101_2_40D = "resnext_101_2x40d"
    RESNEXT_101_4_24D = "resnext_101_4x24d"
    RESNEXT_101_8_14D = "resnext_101_8x14d"
    RESNEXT_101_32_4D = "resnext_101_32x4d"

    SE_RESNET_50 = "se_resnet_50"
    SE_RESNET_101 = "se_resnet_101"
    SE_RESNET_V2_50 = "se_resnet_v2_50"
    SE_RESNET_V2_101 = "se_resnet_v2_101"
    SE_W_RESNET_28_10 = "se_w_resnet_28_10"
    SE_W_RESNET_40_2 = "se_w_resnet_40_2"
    SE_RESNEXT_50_32_4D = "se_resnext_50_32x4d"
    SE_RESNEXT_101_32_4D = "se_resnext_101_32x4d"

    def get_preset(self, preset):
        """Return the parameters dict for a given preset."""
        if isinstance(preset, dict):
            return preset

        match preset:
            case self.RESNET_18:
                return ResNetNetwork.resnet_v1([2, 2, 2, 2], N=2)
            case self.RESNET_34:
                return ResNetNetwork.resnet_v1([3, 4, 6, 3], N=2)
            case self.RESNET_50:
                return ResNetNetwork.resnet_v1([3, 4, 6, 3], N=3)
            case self.RESNET_101:
                return ResNetNetwork.resnet_v1([3, 4, 23, 3], N=3)
            case self.RESNET_152:
                return ResNetNetwork.resnet_v1([3, 8, 36, 3], N=3)
            case self.RESNET_V15_50:
                return ResNetNetwork.resnet_v15([3, 4, 6, 3], N=3)
            case self.RESNET_V15_101:
                return ResNetNetwork.resnet_v15([3, 4, 23, 3], N=3)
            case self.RESNET_V15_152:
                return ResNetNetwork.resnet_v15([3, 8, 36, 3], N=3)
            case self.RESNET_V2_18:
                return ResNetNetwork.resnet_v2([2, 2, 2, 2], N=2)
            case self.RESNET_V2_34:
                return ResNetNetwork.resnet_v2([3, 4, 6, 3], N=2)
            case self.RESNET_V2_50:
                return ResNetNetwork.resnet_v2([3, 4, 6, 3], N=3)
            case self.RESNET_V2_101:
                return ResNetNetwork.resnet_v2([3, 4, 23, 3], N=3)
            case self.RESNET_V2_152:
                return ResNetNetwork.resnet_v2([3, 8, 36, 3], N=3)
            case self.W_RESNET_16_8:
                return ResNetNetwork.w_resnet([2, 2, 2], 8, N=2)
            case self.W_RESNET_16_10:
                return ResNetNetwork.w_resnet([2, 2, 2], 10, N=2)
            case self.W_RESNET_22_8:
                return ResNetNetwork.w_resnet([3, 3, 3], 8, N=2)
            case self.W_RESNET_22_10:
                return ResNetNetwork.w_resnet([3, 3, 3], 10, N=2)
            case self.W_RESNET_28_10:
                return ResNetNetwork.w_resnet([4, 4, 4], 10, N=2)
            case self.W_RESNET_28_12:
                return ResNetNetwork.w_resnet([4, 4, 4], 12, N=2)
            case self.W_RESNET_40_1:
                return ResNetNetwork.w_resnet([6, 6, 6], 1, N=2)
            case self.W_RESNET_40_2:
                return ResNetNetwork.w_resnet([6, 6, 6], 2, N=2)
            case self.W_RESNET_40_4:
                return ResNetNetwork.w_resnet([6, 6, 6], 4, N=2)
            case self.W_RESNET_40_8:
                return ResNetNetwork.w_resnet([6, 6, 6], 8, N=2)
            case self.RESNEXT_29_8_64D:
                return ResNetNetwork.resnext(
                    [3, 3, 3], 8, 64, N=3, expansion_factor=2, base_width=128
                )
            case self.RESNEXT_29_16_64D:
                return ResNetNetwork.resnext(
                    [3, 3, 3], 16, 64, N=3, expansion_factor=2, base_width=128
                )
            case self.RESNEXT_29_32_4D:
                return ResNetNetwork.resnext(
                    [3, 3, 3], 32, 4, N=3, expansion_factor=2, base_width=128
                )
            case self.RESNEXT_50_2_40D:
                return ResNetNetwork.resnext([3, 4, 6, 3], 2, 40, N=3)
            case self.RESNEXT_50_4_24D:
                return ResNetNetwork.resnext([3, 4, 6, 3], 4, 24, N=3)
            case self.RESNEXT_50_8_14D:
                return ResNetNetwork.resnext([3, 4, 6, 3], 8, 14, N=3)
            case self.RESNEXT_50_32_4D:
                return ResNetNetwork.resnext([3, 4, 6, 3], 32, 4, N=3)
            case self.RESNEXT_101_2_40D:
                return ResNetNetwork.resnext([3, 4, 23, 3], 2, 40, N=3)
            case self.RESNEXT_101_4_24D:
                return ResNetNetwork.resnext([3, 4, 23, 3], 4, 24, N=3)
            case self.RESNEXT_101_8_14D:
                return ResNetNetwork.resnext([3, 4, 23, 3], 8, 14, N=3)
            case self.RESNEXT_101_32_4D:
                return ResNetNetwork.resnext([3, 4, 23, 3], 32, 4, N=3)
            case self.SE_RESNET_50:
                return ResNetNetwork.se(ResNetNetwork.resnet_v1([3, 4, 6, 3], N=3))
            case self.SE_RESNET_101:
                return ResNetNetwork.se(ResNetNetwork.resnet_v1([3, 4, 23, 3], N=3))
            case self.SE_RESNET_V2_50:
                return ResNetNetwork.se(ResNetNetwork.resnet_v2([3, 4, 6, 3], N=3))
            case self.SE_RESNET_V2_101:
                return ResNetNetwork.se(ResNetNetwork.resnet_v2([3, 4, 23, 3], N=3))
            case self.SE_W_RESNET_28_10:
                return ResNetNetwork.se(ResNetNetwork.w_resnet([4, 4, 4], 10, N=2))
            case self.SE_W_RESNET_40_2:
                return ResNetNetwork.se(ResNetNetwork.w_resnet([6, 6, 6], 2, N=2))
            case self.SE_RESNEXT_50_32_4D:
                return ResNetNetwork.se(ResNetNetwork.resnext([3, 4, 6, 3], 32, 4, N=3))
            case self.SE_RESNEXT_101_32_4D:
                return ResNetNetwork.se(
                    ResNetNetwork.resnext([3, 4, 23, 3], 32, 4, N=3)
                )

    @staticmethod
    def resnet_v1(s, N=3):
        """Generate parameters for ResNet v1 architecture.

        Parameters
        ----------
            s (list of int): stages sizes e.g. [3, 4, 6, 3] for ResNet-50.
            N (int, optional): Number of convolution layers per residual block,
                defaults = 3. When N=3 kernel sizes are [1, 3, 1] and when N=2 kernel
                sizes are [3, 3].

        Returns
        -------
            dict: parameters for ResNet v1 architecture.
        """
        si = sum(([b] * s[b] for b in range(len(s))), [])
        if N == 2:
            filters = [[2 ** (i + 6), 2 ** (i + 6)] for i in si]
            ks = [3, 3]
        elif N == 3:
            filters = [[2 ** (i + 6), 2 ** (i + 6), 2 ** (i + 8)] for i in si]
            ks = [1, 3, 1]
        else:
            raise ValueError(f"Unsupported N: {N}")

        STRIDE_1, STRIDE_2 = [[1] * N], [[2] + [1] * (N - 1)]
        stride = STRIDE_1 * s[0]  # first block has stride 1
        for b in range(1, len(s)):  # for other blocks : first is 2, rest are 1
            stride += STRIDE_2 + STRIDE_1 * (s[b] - 1)

        act = ["relu"] * N
        nrm = [True] * N

        n = sum(s)
        return {
            "n_residual_blocks": 1 + n,
            "n_conv_per_residual_block": [1] * 1 + [N] * n,
            "n_filters": [[64]] * 1 + filters,
            "kernel_size": [[7]] * 1 + [ks] * n,
            "strides": [[2]] + stride,
            "dilation_rate": 1,
            "padding": "same",
            "activation": [["relu"]] * 1 + [act] * n,
            "use_batch_norm": [[True]] * 1 + [nrm] * n,
            "max_pooling": [True] * 1 + [False] * n,
            "use_residual": [False] * 1 + [True] * n,
            "use_bias": True,
        }

    @staticmethod
    def resnet_v15(s, N=3):
        """Generate parameters for ResNet v1.5 architecture.

        ResNet v1.5 is a variant of ResNet v1 where stride = 2 is applied to the
        3x3 convolution instead of the first 1x1 convolution.

        Parameters
        ----------
            s (list of int): stages sizes e.g. [3, 4, 6, 3] for ResNet-50.
            N (int, optional): Number of convolution layers per residual block,
                defaults = 3. When N=3 kernel sizes are [1, 3, 1] and when N=2 kernel
                sizes are [3, 3].

        Returns
        -------
            dict: parameters for ResNet v1.5 architecture.
        """
        params = ResNetNetwork.resnet_v1(s, N)
        if N == 2:  # when N=2, there is no differences
            return params

        STRIDE_1, STRIDE_2 = [[1] * N], [[1] + [2] + [1] * (N - 2)]
        stride = STRIDE_1 * s[0]  # first block has stride 1
        for b in range(1, len(s)):  # for other blocks : first is 2, rest are 1
            stride += STRIDE_2 + STRIDE_1 * (s[b] - 1)
        params["strides"] = [[2]] + stride
        return params

    @staticmethod
    def resnet_v2(s, N=3):
        """Generate parameters for ResNet v2 architecture.

        ResNet v2 is a ResNet variant that uses pre-activation in the residual blocks.
        It changes the order of operations in the residual block to BN -> ReLU -> Conv,
        instead of Conv -> BN -> ReLU.

        Parameters
        ----------
            s (list of int): stages sizes e.g. [3, 4, 6, 3] for ResNet-50.
            N (int, optional): Number of convolution layers per residual block,
                defaults = 3. When N=3 kernel sizes are [1, 3, 1] and when N=2 kernel
                sizes are [3, 3].

        Returns
        -------
            dict: parameters for ResNet v2 architecture.
        """
        params = ResNetNetwork.resnet_v15(s, N)
        params["resnet_type"] = RESNET_TYPE.V2
        return params

    @staticmethod
    def w_resnet(s, k, N=2):
        """Generate parameters for Wide ResNet architecture.

        Wide ResNet is a variant of ResNet that increases the width of the network by a
        factor of k. It is usurally derived from ResNet v2 architecture, but can be
        derived from other ResNet v1.
        It also adds dropout before the last convolutional layer of each residual block.

        Parameters
        ----------
            s (list of int): stages sizes e.g. [4, 4, 4] for Wide ResNet-28.
            k (int): multiplier for the number of filters in each convolutional layer.
            N (int, optional): Number of convolution layers per residual block,
                defaults = 3. When N=3 kernel sizes are [1, 3, 1] and when N=2 kernel
                sizes are [3, 3].

        Returns
        -------
            dict: parameters for ResNet v2 architecture.
        """
        params = ResNetNetwork.resnet_v2(s, N)

        # multiply filters
        filters = params["n_filters"]
        for b in range(len(filters)):
            for c in range(len(filters[b])):
                filters[b][c] *= k
        params["n_filters"] = filters

        # remove stem kernel size 7 and replace with kernel size 3
        params["kernel_size"][0][0] = 3
        params["strides"][0][0] = 1
        params["max_pooling"][0] = False

        # add dropout
        params["dropout"] = [[0]] * 1 + [[0] * (N - 2) + [0.3, 0]] * (sum(s))
        return params

    @staticmethod
    def resnext(s, g, w, expansion_factor=4, base_width=64, N=3):
        """Generate parameters for ResNeXt architecture.

        ResNeXt is a variant of ResNet that increases the number of paths in each
        residual block by using a technique called "cardinality". It is usually derived
        from ResNet v1 architecture.

        Parameters
        ----------
            s (list of int): stages sizes e.g. [3, 4, 6, 3] for ResNeXt-50.
            g (int): number of groups in each convolutional layer.
            w (int): width of each group.
            expansion_factor (int, optional): factor by which to expand the number of
                filters in the last convolutional layer of each residual block,
                defaults = 4.
            base_width (int, optional): base width for the number of filters in each
                convolutional layer, defaults = 64.
            N (int, optional): Number of convolution layers per residual block,
                defaults = 3. When N=3 kernel sizes are [1, 3, 1] and when N=2 kernel
                sizes are [3, 3].

        Returns
        -------
            dict: parameters for ResNeXt architecture.
        """
        params = ResNetNetwork.resnet_v1(s, N)
        si = sum(([b] * s[b] for b in range(len(s))), [])

        # adjust filters to match ResNeXt paper
        def filters_in(i):
            return base_width * 2**i * g * w // 64

        def filters_out(i):
            return base_width * 2**i * expansion_factor

        if N == 2:
            params["cardinality"] = [[1]] + [[g, g]] * sum(s)
            params["n_filters"] = [[64]] + [[filters_in(i), filters_in(i)] for i in si]
        elif N == 3:
            params["cardinality"] = [[1]] + [[1, g, 1]] * sum(s)
            params["n_filters"] = [[64]] + [
                [filters_in(i), filters_in(i), filters_out(i)] for i in si
            ]
        return params

    @staticmethod
    def se(params):
        """Add Squeeze-and-Excitation (SE) blocks to a ResNet preset.

        Parameters
        ----------
            params (dict): parameters for a ResNet architecture.

        Returns
        -------
            dict: parameters for a ResNet architecture with SE blocks.
        """
        params["se_ratio"] = 16
        return params
