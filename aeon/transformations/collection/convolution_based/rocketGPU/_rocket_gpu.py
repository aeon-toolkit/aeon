"""Rocket transformer for GPU."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["ROCKETGPU"]

import numpy as np

# Import CPU's kernel generation function
from aeon.transformations.collection.convolution_based._rocket import (
    _generate_kernels,
)
from aeon.transformations.collection.convolution_based.rocketGPU.base import (
    BaseROCKETGPU,
)


class ROCKETGPU(BaseROCKETGPU):
    """RandOm Convolutional KErnel Transform (ROCKET) for GPU.

    A kernel (or convolution) is a subseries used to create features that can be used
    in machine learning tasks. ROCKET [1]_ generates a large number of random
    convolutional kernels in the fit method. The length and dilation of each kernel
    are also randomly generated. The kernels are used in the transform stage to
    generate a new set of features. A kernel is used to create an activation map for
    each series by running it across a time series, including random length and
    dilation. It transforms the time series with two features per kernel. The first
    feature is global max pooling and the second is proportion of positive values
    (or PPV).


    Parameters
    ----------
    n_kernels : int, default=10000
       Number of random convolutional filters.
    batch_size : int, default = 64
        The batch to parallelize over GPU.
    random_state : None or int, optional, default = None
        Seed for random number generation.

    Notes
    -----
    This GPU implementation uses the CPU's kernel generation logic
    (from `_rocket._generate_kernels`) to ensure exact kernel parity
    when using the same random seed.

    References
    ----------
    .. [1] Tan, Chang Wei and Dempster, Angus and Bergmeir, Christoph
        and Webb, Geoffrey I,
        "ROCKET: Exceptionally fast and accurate time series
      classification using random convolutional kernels",2020,
      https://link.springer.com/article/10.1007/s10618-020-00701-z,
      https://arxiv.org/abs/1910.13051
    """

    def __init__(
        self,
        n_kernels=10000,
        batch_size=64,
        random_state=None,
    ):
        super().__init__(n_kernels)

        self.n_kernels = n_kernels
        self.batch_size = batch_size
        self.random_state = random_state

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        self
        """
        self.input_length = X.shape[2]
        self.n_channels = X.shape[1]

        self.kernels = _generate_kernels(
            n_timepoints=self.input_length,
            n_kernels=self.n_kernels,
            n_channels=self.n_channels,
            seed=self.random_state,
        )
        self._convert_cpu_kernels_to_gpu_format()
        return self

    def _convert_cpu_kernels_to_gpu_format(self):
        """Convert CPU's kernel format to GPU's TensorFlow-compatible format.

        CPU kernels are stored compactly as:
        (weights,lengths,biases,dilations,paddings,num_channel_indices,channel_indices)

        GPU needs:
        - _list_of_kernels: List of (kernel_length, n_channels, 1) arrays
        - _list_of_dilations: List of int dilation rates
        - _list_of_paddings: List of "SAME" or "VALID" strings
        - _list_of_biases: List of float bias values

        The key conversion is handling CPU's selective channel indexing
        by creating dense kernels with zero weights for unused channels.
        """
        (
            weights,
            lengths,
            biases,
            dilations,
            paddings,
            num_channel_indices,
            channel_indices,
        ) = self.kernels

        self._list_of_kernels = []
        self._list_of_dilations = []
        self._list_of_paddings = []
        self._list_of_biases = []

        weight_idx = 0
        channel_idx = 0

        for i in range(self.n_kernels):
            kernel_length = lengths[i]
            n_kernel_channels = num_channel_indices[i]

            # Extract this kernel's sparse weights from CPU format
            n_weights = kernel_length * n_kernel_channels
            sparse_weights = weights[weight_idx : weight_idx + n_weights]
            sparse_weights = sparse_weights.reshape((n_kernel_channels, kernel_length))

            # Get which channels this kernel operates on
            selected_channels = channel_indices[
                channel_idx : channel_idx + n_kernel_channels
            ]

            # Create dense kernel tensor: (kernel_length, n_channels, 1)
            # Unused channels have zero weights (no contribution to convolution)
            dense_kernel = np.zeros(
                (kernel_length, self.n_channels, 1), dtype=np.float32
            )

            # Place sparse weights in the corresponding channel positions
            # Preserving the exact channel order from CPU
            for c_idx, channel in enumerate(selected_channels):
                dense_kernel[:, channel, 0] = sparse_weights[c_idx, :]

            self._list_of_kernels.append(dense_kernel)

            # Convert numeric padding to TensorFlow categorical padding
            # CPU: 0 or (length-1)*dilation//2
            # GPU: "VALID" or "SAME"
            if paddings[i] == 0:
                self._list_of_paddings.append("VALID")
            else:
                # Non-zero padding -> use SAME to approximate symmetric padding
                self._list_of_paddings.append("SAME")

            # Convert dilation and bias to Python scalar types
            self._list_of_dilations.append(int(dilations[i]))
            self._list_of_biases.append(float(biases[i]))

            # Advance indices for next kernel
            weight_idx += n_weights
            channel_idx += n_kernel_channels

    def _generate_batch_indices(self, n):
        """Generate the list of batches.

        Parameters
        ----------
        n : int
            The number of samples in the dataset.

        Returns
        -------
        batch_indices_list : list
            A list of multiple np.ndarray containing indices of batches.
        """
        import numpy as np

        all_indices = np.arange(n)

        if self.batch_size >= n:
            return [all_indices]

        remainder_batch_size = n % self.batch_size
        number_batches = n // self.batch_size

        batch_indices_list = np.array_split(
            ary=all_indices[: n - remainder_batch_size],
            indices_or_sections=number_batches,
        )

        if remainder_batch_size > 0:
            batch_indices_list.append(all_indices[n - remainder_batch_size :])

        return batch_indices_list

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
            collection of time series to transform.
        y : ignored argument for interface compatibility.

        Returns
        -------
        output_rocket : np.ndarray [n_cases, n_kernels * 2]
            transformed features.
        """
        import tensorflow as tf

        tf.random.set_seed(self.random_state)

        # Transpose and convert to float32 for TensorFlow compatibility
        X = X.transpose(0, 2, 1).astype(np.float32)

        batch_indices_list = self._generate_batch_indices(n=len(X))

        output_features = []

        for f in range(self.n_kernels):
            output_features_filter = []

            for batch_indices in batch_indices_list:
                _output_convolution = tf.nn.conv1d(
                    input=X[batch_indices],
                    stride=1,
                    filters=self._list_of_kernels[f],
                    dilations=self._list_of_dilations[f],
                    padding=self._list_of_paddings[f],
                )

                _output_convolution = np.squeeze(_output_convolution.numpy(), axis=-1)
                _output_convolution += self._list_of_biases[f]

                _ppv = self._get_ppv(x=_output_convolution)
                _max = self._get_max(x=_output_convolution)

                output_features_filter.append(
                    np.concatenate(
                        (np.expand_dims(_ppv, axis=-1), np.expand_dims(_max, axis=-1)),
                        axis=1,
                    )
                )

            output_features.append(
                np.expand_dims(np.concatenate(output_features_filter, axis=0), axis=0)
            )

        output_rocket = np.concatenate(output_features, axis=0).swapaxes(0, 1)
        output_rocket = output_rocket.reshape(
            (output_rocket.shape[0], output_rocket.shape[1] * output_rocket.shape[2])
        )

        return output_rocket

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the transformer.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {
            "n_kernels": 5,
        }
        return params
