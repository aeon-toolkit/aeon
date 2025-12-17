"""Rocket transformer for GPU."""

__maintainer__ = ["hadifawaz1999"]
__all__ = ["ROCKETGPU"]

import numpy as np

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
    kernel_size : list, default = None
        The list of possible kernel sizes, default is [7, 9, 11].
    padding : list, default = None
        The list of possible tensorflow padding, default is ["SAME", "VALID"].
    use_dilation : bool, default = True
        Whether or not to use dilation in convolution operations.
    bias_range : Tuple, default = None
        The min and max value of bias values, default is (-1.0, 1.0).
    batch_size : int, default = 64
        The batch to parallelize over GPU.
    random_state : None or int, optional, default = None
        Seed for random number generation.

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
        kernel_size=None,
        padding=None,
        use_dilation=True,
        bias_range=None,
        batch_size=64,
        random_state=None,
    ):
        super().__init__(n_kernels)

        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_dilation = use_dilation
        self.bias_range = bias_range
        self.batch_size = batch_size
        self.random_state = random_state

    def _define_parameters(self):
        """Define the parameters of ROCKET."""
        rng = np.random.default_rng(self.random_state)

        self._list_of_kernels = []
        self._list_of_dilations = []
        self._list_of_paddings = []
        self._list_of_biases = []

        for _ in range(self.n_kernels):
            _kernel_size = rng.choice(self._kernel_size, size=1)[0]
            _convolution_kernel = rng.normal(size=(_kernel_size, self.n_channels, 1))
            _convolution_kernel = _convolution_kernel - _convolution_kernel.mean(
                axis=0, keepdims=True
            )

            if self.use_dilation:
                _dilation_rate = 2 ** rng.uniform(
                    0, np.log2((self.input_length - 1) / (_kernel_size - 1))
                )
            else:
                _dilation_rate = 1

            _padding = rng.choice(self._padding, size=1)[0]
            assert _padding in ["SAME", "VALID"]

            _bias = rng.uniform(self._bias_range[0], self._bias_range[1])

            self._list_of_kernels.append(_convolution_kernel)
            self._list_of_dilations.append(_dilation_rate)
            self._list_of_paddings.append(_padding)
            self._list_of_biases.append(_bias)

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

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

        self._kernel_size = [7, 9, 11] if self.kernel_size is None else self.kernel_size
        self._padding = ["VALID", "SAME"] if self.padding is None else self.padding
        self._bias_range = (-1.0, 1.0) if self.bias_range is None else self.bias_range

        assert self._bias_range[0] <= self._bias_range[1]

        self._define_parameters()

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

        X = X.transpose(0, 2, 1)

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
