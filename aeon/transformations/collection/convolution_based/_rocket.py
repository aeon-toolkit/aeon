"""Rocket transformer."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["Rocket"]

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from aeon.transformations.collection import BaseCollectionTransformer, Normalizer
from aeon.utils.validation import check_n_jobs

# Renaming existing Rocket class to _RocketCPU (internal CPU implementation)


class _RocketCPU(BaseCollectionTransformer):
    """Internal CPU implementation of ROCKET using Numba.

    This class contains the original CPU-based implementation. Users should not
    instantiate this directly - use the unified Rocket class instead.

    Note: This is the same implementation as the original Rocket class,
    just renamed to distinguish it from the new unified wrapper.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_kernels=10_000,
        normalise=True,
        n_jobs=1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.normalise = normalise
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y=None):
        """Generate random kernels adjusted to time series shape.

        Infers time series length and number of channels from input numpy array,
        and generates random kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        self
        """
        self._n_jobs = check_n_jobs(self.n_jobs)

        if isinstance(self.random_state, int):
            self._random_state = self.random_state
        else:
            self._random_state = None
        n_channels = X[0].shape[0]

        # The only use of n_timepoints is to set the maximum dilation
        self.fit_min_length_ = X[0].shape[1]
        self.kernels = _generate_kernels(
            self.fit_min_length_, self.n_kernels, n_channels, self._random_state
        )
        return self

    def _transform(self, X, y=None):
        """Transform input time series using random convolutional kernels.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            collection of time series to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        np.ndarray (n_cases, n_kernels), transformed features
        """
        if self.normalise:
            norm = Normalizer()
            X = norm.fit_transform(X)

        prev_threads = get_num_threads()
        set_num_threads(self._n_jobs)

        X_ = _apply_kernels(X, self.kernels)

        set_num_threads(prev_threads)
        return X_


class Rocket(BaseCollectionTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET) with unified CPU/GPU interface.

    ROCKET generates random convolutional kernels to transform time series data into
    features for machine learning. This unified interface allows you to choose between
    CPU (using Numba) or GPU (using TensorFlow) execution via the `device` parameter.

    A kernel (convolution) is a pattern detector that slides across your time series.
    ROCKET creates thousands of these random kernels during fitting, then uses them
    to extract meaningful features during transformation. Each kernel produces two
    features: the maximum activation (global max pooling) and the proportion of
    positive values (PPV).

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of random convolutional kernels to generate. More kernels typically
        improve accuracy but increase computation time. Both CPU and GPU versions
        use this parameter.

    device : str, default="cpu"
        Computation device to use for transformation. Options:

        - "cpu" : Use CPU with Numba JIT compilation (default, most compatible)
        - "gpu" : Use GPU with TensorFlow (requires TensorFlow and GPU hardware)
        - "auto" : Automatically detect GPU availability and use it if possible,
                   otherwise fall back to CPU with a warning

    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility. Note that CPU and GPU implementations
        use different random number generation methods, so the same seed will
        produce different (but equally valid) kernels on each device.

    normalise : bool, default=True
        [CPU ONLY] Whether to normalize input time series to zero mean and unit
        variance before transformation. Ignored when device="gpu".

    n_jobs : int, default=1
        [CPU ONLY] Number of parallel threads to use. Set to -1 to use all available
        processors. Ignored when device="gpu" (GPU handles parallelization internally).

    kernel_size : list of int or None, default=None
        [GPU ONLY] Allowed kernel sizes to randomly choose from. If None, defaults
        to [7, 9, 11]. Ignored when device="cpu".

    padding : list of str or None, default=None
        [GPU ONLY] Padding strategies for convolution. If None, defaults to
        ["SAME", "VALID"]. Ignored when device="cpu".

    use_dilation : bool, default=True
        [GPU ONLY] Whether to apply random dilation to kernels. Ignored when
        device="cpu" (CPU always uses dilation).

    bias_range : tuple of float or None, default=None
        [GPU ONLY] Range for random bias initialization as (min, max). If None,
        defaults to (-1.0, 1.0). Ignored when device="cpu".

    batch_size : int, default=64
        [GPU ONLY] Number of samples to process per GPU batch. Larger values may
        be faster but require more GPU memory. Ignored when device="cpu".

    Attributes
    ----------
    _impl : _RocketCPU or ROCKETGPU
        The actual implementation object (CPU or GPU) that performs the work.
        This is set during fitting based on the device parameter.

    _actual_device : str
        The device that was actually selected after considering availability.
        For device="auto", this shows which device was chosen.

    See Also
    --------
    MiniRocket : Faster variant using fixed kernels
    MultiRocket : Enhanced variant with multiple pooling operations

    Notes
    -----
    **Device Selection:**
    - CPU version is always available and works on any system
    - GPU version requires TensorFlow installation and compatible GPU hardware
    - Using device="auto" is recommended for portable code

    **Pickling:**
    - CPU version can be saved with pickle (portable across systems)
    - GPU version cannot be pickled due to TensorFlow limitations
    - If you need to save a GPU model, save the transformed features instead

    **Performance:**
    - GPU is typically faster for large datasets (1000+ samples)
    - CPU may be faster for small datasets due to GPU overhead
    - CPU is more memory efficient for very long time series

    References
    ----------
    .. [1] Dempster, A., Petitjean, F., & Webb, G. I. (2020).
       ROCKET: exceptionally fast and accurate time series classification using
       random convolutional kernels. Data Mining and Knowledge Discovery, 34(5).

    Examples
    --------
    Basic usage with CPU (default):

    >>> from aeon.transformations.collection.convolution_based import Rocket
    >>> from aeon.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train")
    >>> X_test, y_test = load_unit_test(split="test")
    >>>
    >>> # Default CPU execution
    >>> rocket = Rocket(n_kernels=1000)
    >>> rocket.fit(X_train)
    Rocket(...)
    >>> X_train_features = rocket.transform(X_train)
    >>> X_test_features = rocket.transform(X_test)

    Using GPU explicitly:

    >>> # Requires TensorFlow and GPU
    >>> rocket_gpu = Rocket(n_kernels=1000, device="gpu")
    >>> rocket_gpu.fit(X_train)
    >>> X_train_features = rocket_gpu.transform(X_train)

    Auto-detection with fallback:

    >>> # Will use GPU if available, otherwise CPU
    >>> rocket_auto = Rocket(n_kernels=1000, device="auto")
    >>> rocket_auto.fit(X_train)
    >>> # Check which device was used
    >>> print(f"Using: {rocket_auto._actual_device}")

    CPU-specific parameters:

    >>> # Normalize data and use 4 CPU cores
    >>> rocket_cpu = Rocket(
    ...     n_kernels=1000,
    ...     device="cpu",
    ...     normalise=True,
    ...     n_jobs=4
    ... )

    GPU-specific parameters:

    >>> # Custom kernel sizes and batch processing
    >>> rocket_gpu = Rocket(
    ...     n_kernels=1000,
    ...     device="gpu",
    ...     kernel_size=[7, 9, 11, 13],
    ...     batch_size=128
    ... )
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,  # Has n_jobs parameter
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
        # Note: cant_pickle tag is set dynamically during _fit for GPU
    }

    def __init__(
        self,
        n_kernels=10_000,
        device="cpu",
        random_state=None,
        # CPU-only parameters
        normalise=True,
        n_jobs=1,
        # GPU-only parameters
        kernel_size=None,
        padding=None,
        use_dilation=True,
        bias_range=None,
        batch_size=64,
    ):
        # Common parameters (work for both CPU and GPU)
        self.n_kernels = n_kernels
        self.device = device
        self.random_state = random_state

        # CPU-only parameters (ignored when device="gpu")
        self.normalise = normalise
        self.n_jobs = n_jobs

        # GPU-only parameters (ignored when device="cpu")
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_dilation = use_dilation
        self.bias_range = bias_range
        self.batch_size = batch_size

        # These will be set during fit
        self._impl = None  # The actual CPU or GPU implementation object
        self._actual_device = None  # The device actually being used

        super().__init__()

    #  GPU Detection Method

    def _detect_gpu_available(self):
        """Check if GPU and TensorFlow are available on this system.

        This method tries to import TensorFlow and query for GPU devices.
        It returns True only if both TensorFlow is installed AND at least
        one GPU device is detected.

        Returns
        -------
        bool
            True if TensorFlow is installed and GPU is available, False otherwise.
        """
        try:
            # Try importing TensorFlow (will fail if not installed)
            import tensorflow as tf

            # Ask TensorFlow to list all physical GPU devices
            gpus = tf.config.list_physical_devices("GPU")

            # Return True only if we found at least one GPU
            return len(gpus) > 0

        except ImportError:
            # TensorFlow is not installed - GPU not available
            return False
        except Exception:
            # Some other error (maybe TensorFlow is broken) - treat as unavailable
            return False

    #  GPU/CPU Device Selection Logic

    def _select_device(self):
        """Determine which device to use based on preference and availability.

        This method handles three modes:
        - "cpu": Always use CPU (safe and predictable)
        - "gpu": Force GPU usage (error if unavailable)
        - "auto": Try GPU first, fallback to CPU if needed

        Returns
        -------
        str
            Either "cpu" or "gpu" - the device that will actually be used.

        Raises
        ------
        RuntimeError
            If device="gpu" but GPU/TensorFlow is not available.
        ValueError
            If device parameter has an invalid value.
        """
        if self.device == "cpu":
            # User explicitly wants CPU - no need to check anything
            return "cpu"

        elif self.device == "gpu":
            # User explicitly wants GPU - make sure it's available
            if not self._detect_gpu_available():
                raise RuntimeError(
                    "GPU device requested but not available. "
                    "Please ensure TensorFlow is installed with GPU support and "
                    "that compatible GPU hardware is accessible. "
                    "\n\nTo fix this:\n"
                    "1. Install TensorFlow: pip install tensorflow\n"
                    "2. Ensure you have a compatible NVIDIA GPU\n"
                    "3. Install CUDA and cuDNN libraries\n"
                    "\nAlternatively, use device='cpu' or device='auto' "
                    "for automatic fallback."
                )
            return "gpu"

        elif self.device == "auto":
            # Try GPU, fallback to CPU if not available
            if self._detect_gpu_available():
                # GPU is available - inform user we're using it
                import warnings

                warnings.warn(
                    "GPU detected and will be used for ROCKET transformation. "
                    "This typically provides better performance for large datasets. "
                    "To explicitly use CPU instead, set device='cpu'.",
                    UserWarning,
                    stacklevel=2,
                )
                return "gpu"
            else:
                # GPU not available - fallback to CPU with informative warning
                import warnings

                warnings.warn(
                    "GPU not detected or TensorFlow not installed. "
                    "Falling back to CPU execution. "
                    "To suppress this warning, explicitly set device='cpu'. "
                    "To use GPU, install TensorFlow with GPU support.",
                    UserWarning,
                    stacklevel=2,
                )
                return "cpu"

        else:
            # Invalid device parameter - help user fix it
            raise ValueError(
                f"Invalid device '{self.device}'. "
                f"Expected 'cpu', 'gpu', or 'auto', but got '{self.device}'. "
                f"Please use one of the valid options."
            )

    # Parameter Validation with Warnings - notify if parameters are
    # not compatible with selected device

    def _validate_parameters(self, actual_device):
        """Warn if parameters won't work with selected device.

        This helps prevent confusion when users accidentally set parameters that
        only work on a different device. For example, setting normalise=False when
        using GPU (which doesn't support normalization anyway).

        Parameters
        ----------
        actual_device : str
            The device that was selected ("cpu" or "gpu").
        """
        import warnings

        if actual_device == "cpu":
            # Check if user provided GPU-only parameters that will be ignored
            gpu_params_provided = []

            # Check each GPU-only parameter to see if it differs from default
            if self.kernel_size is not None:
                gpu_params_provided.append("kernel_size")
            if self.padding is not None:
                gpu_params_provided.append("padding")
            if self.use_dilation is not True:  # True is the default
                gpu_params_provided.append("use_dilation")
            if self.bias_range is not None:
                gpu_params_provided.append("bias_range")
            if self.batch_size != 64:  # 64 is the default
                gpu_params_provided.append("batch_size")

            # If any GPU parameters were set, warn the user
            if gpu_params_provided:
                warnings.warn(
                    f"Device is set to 'cpu', but GPU-only parameters were provided: "
                    f"{', '.join(gpu_params_provided)}. "
                    f"These parameters will be IGNORED during execution. "
                    f"To use these parameters, set device='gpu'. "
                    f"To suppress this warning, remove the GPU-only parameters.",
                    UserWarning,
                    stacklevel=4,  # Show warning at user's code level
                )

        elif actual_device == "gpu":
            # Check if user provided CPU-only parameters that will be ignored
            cpu_params_provided = []

            # Check each CPU-only parameter to see if it differs from default
            if self.normalise is not True:  # True is the default
                cpu_params_provided.append("normalise")
            if self.n_jobs != 1:  # 1 is the default
                cpu_params_provided.append("n_jobs")

            # If any CPU parameters were set, warn the user
            if cpu_params_provided:
                warnings.warn(
                    f"Device is set to 'gpu', but CPU-only parameters were provided: "
                    f"{', '.join(cpu_params_provided)}. "
                    f"These parameters will be IGNORED during execution. "
                    f"To use these parameters, set device='cpu'. "
                    f"To suppress this warning, remove the CPU-only parameters.",
                    UserWarning,
                    stacklevel=4,  # Show warning at user's code level
                )

    # Delegation in _fit Method - ensure model fits on correct device
    # and parameters are validated

    def _fit(self, X, y=None):
        """Fit the ROCKET transformer by generating random kernels.

        This is where the magic happens - we decide whether to use CPU or GPU,
        create the appropriate implementation object, warn about any parameter
        issues, and then delegate the actual fitting work to that implementation.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The training time series data.
        y : ignored
            Exists for API compatibility but is not used.

        Returns
        -------
        self
            The fitted transformer, ready to transform new data.
        """
        self._actual_device = self._select_device()
        self._validate_parameters(self._actual_device)
        # Create the appropriate implementation (CPU or GPU)
        if self._actual_device == "cpu":
            # User needs CPU - create internal CPU implementation
            self._impl = _RocketCPU(
                n_kernels=self.n_kernels,
                normalise=self.normalise,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )

        else:  # self._actual_device == "gpu"
            # User needs GPU - import and create GPU implementation
            # We only import when needed to avoid requiring TensorFlow for CPU users
            from .rocketGPU._rocket_gpu import ROCKETGPU

            self._impl = ROCKETGPU(
                n_kernels=self.n_kernels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                use_dilation=self.use_dilation,
                bias_range=self.bias_range,
                batch_size=self.batch_size,
                random_state=self.random_state,
            )
        self._impl._fit(X, y)
        self._update_tags_for_device()
        return self

    def _update_tags_for_device(self):
        """Update class tags based on which device is being used.

        Different devices have different capabilities:
        - CPU can be pickled, supports multithreading, no external dependencies
        - GPU cannot be pickled, no multithreading (GPU handles it), requires TensorFlow
        """
        if self._actual_device == "gpu":
            # GPU version has special limitations
            self.set_tags(
                **{
                    "cant_pickle": True,  # TensorFlow objects can't be pickled
                    "python_dependencies": "tensorflow",  # Requires TensorFlow
                    "capability:multithreading": False,  # GPU handles parallelization
                }
            )
        else:
            # CPU version is more flexible
            self.set_tags(
                **{
                    "capability:multithreading": True,  # Can use multiple threads
                    # No special pickling or dependency requirements
                }
            )

    # Delegation in _transform Method

    def _transform(self, X, y=None):
        """Transform time series using the fitted kernels.

        This method simply delegates to whichever implementation (CPU or GPU)
        was selected during fitting. The implementation object already knows
        what to do.

        Parameters
        ----------
        X : 3D np.ndarray of shape (n_cases, n_channels, n_timepoints)
            The time series data to transform.
        y : ignored
            Exists for API compatibility but is not used.

        Returns
        -------
        np.ndarray of shape (n_cases, n_kernels * 2)
            Transformed features. Each kernel produces 2 features (PPV and max).

        Raises
        ------
        ValueError
            If transform is called before fit.
        """
        # Make sure user called fit() first
        if self._impl is None:
            raise ValueError(
                "This Rocket instance is not fitted yet. "
                "Call 'fit' with appropriate data before using 'transform'."
            )

        # Delegate transformation to the implementation
        return self._impl._transform(X, y)

    # Difference-Handling Pickling Limitations

    def __getstate__(self):
        """Control what happens when someone tries to pickle this object.

        CPU version can be pickled normally (Numba and numpy are pickle-friendly).
        GPU version cannot be pickled due to TensorFlow's complex state (GPU pointers,
        computational graphs, etc. that can't be serialized).

        This method is automatically called by Python's pickle module.

        Returns
        -------
        dict
            Object state to pickle (for CPU version).

        Raises
        ------
        TypeError
            If trying to pickle GPU version (not supported).
        """
        if self._actual_device == "gpu":
            # GPU version cannot be pickled - give user helpful error
            raise TypeError(
                "Cannot pickle Rocket transformer with device='gpu' due to "
                "TensorFlow limitations. TensorFlow's GPU state (computational graphs, "
                "GPU memory pointers, CUDA contexts) cannot be serialized.\n\n"
                "Solutions:\n"
                "1. Use device='cpu' instead (CPU version can be pickled)\n"
                "2. Save only the transformed features with np.save()\n"
                "3. Re-fit the transformer when needed (fitting is usually fast)\n"
                "4. For deployment, transform data once and save features\n\n"
                "Example - Save features instead:\n"
                "  X_features = rocket.transform(X)\n"
                "  np.save('features.npy', X_features)"
            )

        # CPU version - pickle normally
        return self.__dict__

    def __setstate__(self, state):
        """Restore object from pickle (CPU version only).

        This method is automatically called by Python's pickle module when
        unpickling (loading) a saved object.

        Parameters
        ----------
        state : dict
            The saved object state.
        """
        # Restore all instance variables from saved state
        self.__dict__.update(state)


@njit(fastmath=True, cache=True)
def _generate_kernels(n_timepoints, n_kernels, n_channels, seed):
    if seed is not None:
        np.random.seed(seed)
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, n_kernels).astype(np.int32)

    num_channel_indices = np.zeros(n_kernels, dtype=np.int32)
    for i in range(n_kernels):
        limit = min(n_channels, lengths[i])
        num_channel_indices[i] = 2 ** np.random.uniform(0, np.log2(limit + 1))

    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros(
        np.int32(
            np.dot(lengths.astype(np.float32), num_channel_indices.astype(np.float32))
        ),
        dtype=np.float32,
    )
    biases = np.zeros(n_kernels, dtype=np.float32)
    dilations = np.zeros(n_kernels, dtype=np.int32)
    paddings = np.zeros(n_kernels, dtype=np.int32)

    a1 = 0  # for weights
    a2 = 0  # for channel_indices

    for i in range(n_kernels):
        _length = lengths[i]
        _num_channel_indices = num_channel_indices[i]

        _weights = np.random.normal(0, 1, _num_channel_indices * _length).astype(
            np.float32
        )

        b1 = a1 + (_num_channel_indices * _length)
        b2 = a2 + _num_channel_indices

        a3 = 0  # for weights (per channel)
        for _ in range(_num_channel_indices):
            b3 = a3 + _length
            _weights[a3:b3] = _weights[a3:b3] - _weights[a3:b3].mean()
            a3 = b3

        weights[a1:b1] = _weights

        channel_indices[a2:b2] = np.random.choice(
            np.arange(0, n_channels), _num_channel_indices, replace=False
        )

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(
            0, np.log2((n_timepoints - 1) / (_length - 1))
        )
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

        a1 = b1
        a2 = b2

    return (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        num_channel_indices,
        channel_indices,
    )


@njit(
    parallel=True,
    fastmath=True,
    cache=True,
)
def _apply_kernels(X, kernels):
    (
        weights,
        lengths,
        biases,
        dilations,
        paddings,
        n_channel_indices,
        channel_indices,
    ) = kernels
    n_cases = len(X)
    n_channels, _ = X[0].shape
    n_kernels = len(lengths)

    _X = np.zeros((n_cases, n_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(n_cases):
        a1 = 0  # for weights
        a2 = 0  # for channel_indices
        a3 = 0  # for features

        for j in range(n_kernels):
            b1 = a1 + n_channel_indices[j] * lengths[j]
            b2 = a2 + n_channel_indices[j]
            b3 = a3 + 2

            if n_channel_indices[j] == 1:
                _X[i][a3:b3] = _apply_kernel_univariate(
                    X[i][channel_indices[a2]],
                    weights[a1:b1],
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                )

            else:
                _weights = weights[a1:b1].reshape((n_channel_indices[j], lengths[j]))

                _X[i][a3:b3] = _apply_kernel_multivariate(
                    X[i],
                    _weights,
                    lengths[j],
                    biases[j],
                    dilations[j],
                    paddings[j],
                    n_channel_indices[j],
                    channel_indices[a2:b2],
                )

            a1 = b1
            a2 = b2
            a3 = b3

    return _X.astype(np.float32)


@njit(fastmath=True, cache=True)
def _apply_kernel_univariate(X, weights, length, bias, dilation, padding):
    """Apply a single kernel to a univariate series."""
    n_timepoints = len(X)

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf

    end = (n_timepoints + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias

        index = i

        for j in range(length):
            if index > -1 and index < n_timepoints:
                _sum = _sum + weights[j] * X[index]

            index = index + dilation

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return np.float32(_ppv / output_length), np.float32(_max)


@njit(fastmath=True, cache=True)
def _apply_kernel_multivariate(
    X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices
):
    """Apply a kernel to a single multivariate time series."""
    n_columns, n_timepoints = X.shape

    output_length = (n_timepoints + (2 * padding)) - ((length - 1) * dilation)

    _ppv = 0
    _max = -np.inf
    end = (n_timepoints + padding) - ((length - 1) * dilation)
    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if index > -1 and index < n_timepoints:
                for k in range(num_channel_indices):
                    _sum = _sum + weights[k, j] * X[channel_indices[k], index]
            index = index + dilation
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1
    return np.float32(_ppv / output_length), np.float32(_max)
