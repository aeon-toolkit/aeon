"""Signature transformer."""

import collections
import math

import numpy as np
from sklearn.pipeline import Pipeline

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.signature_based._augmentations import (
    _make_augmentation_pipeline,
)


class SignatureTransformer(BaseCollectionTransformer):
    """Transformation class from the signature method.

    Follows the methodology laid out in the paper:
        "A Generalised Signature Method for Multivariate Time Series"

    Parameters
    ----------
    augmentation_list : tuple of strings, default = ``("basepoint", "addtime")``
        Contains the augmentations to be applied before application of the signature
        transform.
    window_name : str, default="dyadic"
        The name of the window transform to apply.
    window_depth : int, default=3
        The depth of the dyadic window. (Active only if ``window_name == 'dyadic'``).
    window_length : int or None, default=None
        The length of the sliding/expanding window. (Active only if ``window_name``
        in ``['sliding, 'expanding']``.
    window_step : int or None, default=None
        The step of the sliding/expanding window. (Active only if ``window_name in [
        'sliding, 'expanding']``.
    rescaling : str or None, default=None
        The method of signature rescaling.
    sig_tfm : str, default="signature
        Specifies the type of signature transform. One of:
        ``['signature', 'logsignature']``).
    depth: int, default=4
        Signature truncation depth.

    Attributes
    ----------
    self.signature_method_ : sklearn.Pipeline
        sklearn pipeline object that contains all the steps to extract the signature
        features.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "python_dependencies": "esig",
    }

    def __init__(
        self,
        augmentation_list=("basepoint", "addtime"),
        window_name: str = "dyadic",
        window_depth: int = 3,
        window_length=None,
        window_step=None,
        rescaling=None,
        sig_tfm: str = "signature",
        depth: int = 4,
    ):
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.signature_method_ = None

        super().__init__()

    def _fit(self, X, y=None):
        """Set up the signature method as an sklearn pipeline."""
        augmentation_step = _make_augmentation_pipeline(self.augmentation_list)
        transform_step = _WindowSignatureTransform(
            window_name=self.window_name,
            window_depth=self.window_depth,
            window_length=self.window_length,
            window_step=self.window_step,
            sig_tfm=self.sig_tfm,
            sig_depth=self.depth,
            rescaling=self.rescaling,
        )

        # 'signature method' as defined in the reference paper
        self.signature_method_ = Pipeline(
            [
                ("augmentations", augmentation_step),
                ("window_and_transform", transform_step),
            ]
        )
        self.signature_method_.fit(X)
        return self

    def _transform(self, X, y=None):
        return self.signature_method_.transform(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

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
            "augmentation_list": ("basepoint", "addtime"),
            "depth": 3,
            "window_name": "global",
        }
        return params


class _WindowSignatureTransform(BaseCollectionTransformer):
    """Perform the signature transform over given windows.

    Input series Given data of shape (n_N, L, C] and specification of a window method
    from the
    signatures window module, this class will compute the signatures over
    each window (for the given signature options) and concatenate the results
    into a tensor of shape [N, num_sig_features * num_windows].

    Parameters
    ----------
    num_intervals: int, dimension of the transformed data (default 8)
    """

    _tags = {
        "fit_is_empty": True,
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "python_dependencies": "esig",
    }

    def __init__(
        self,
        window_name=None,
        window_depth=None,
        window_length=None,
        window_step=None,
        sig_tfm=None,
        sig_depth=None,
        rescaling=None,
    ):
        super().__init__()
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.sig_tfm = sig_tfm
        self.sig_depth = sig_depth
        self.rescaling = rescaling

        self.window = _window_getter(
            self.window_name, self.window_depth, self.window_length, self.window_step
        )

    def _transform(self, X, y=None):
        import esig

        # monkey patch due to bug in esig which causes some data shapes to crash.
        # Remove if it is fixed upstream I guess.
        def prepare_stream(self, stream_data, depth):
            import numpy as np
            import roughpy as rp

            no_samples, width = stream_data.shape
            increments = np.diff(stream_data, axis=0)
            indices = np.arange(no_samples - 1) / (no_samples - 1)
            context = rp.get_context(width, depth, rp.DPReal)
            stream = rp.LieIncrementStream.from_increments(
                increments, indices=indices, ctx=context
            )
            return stream

        esig.backends.RoughPyBackend.prepare_stream = prepare_stream

        depth = self.sig_depth
        data = np.swapaxes(X, 1, 2)

        # Path rescaling
        if self.rescaling == "pre":
            data = _rescale_path(data, depth)

        # Prepare for signature computation
        if self.sig_tfm == "signature":

            def transform(x):
                return esig.stream2sig(x, depth)[1:].reshape(-1, 1)

        else:

            def transform(x):
                return esig.stream2logsig(x, depth).reshape(1, -1)

        length = data.shape[1]

        # Compute signatures in each window returning the grouped structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                # Signature computation step
                signature = np.stack(
                    [transform(x[window.start : window.end]) for x in data]
                ).reshape(data.shape[0], -1)
                # Rescale if specified
                if self.rescaling == "post":
                    signature = _rescale_signature(signature, data.shape[2], depth)

                signature_group.append(signature)
            signatures.append(signature_group)

        # We are currently not considering deep models and so return all the
        # features concatenated together
        signatures = np.concatenate([x for lst in signatures for x in lst], axis=1)

        return signatures


_Pair = collections.namedtuple("Pair", ("start", "end"))


def _window_getter(
    window_name, window_depth=None, window_length=None, window_step=None
):
    """Window getter.

    Gets the window method correspondent to the given string and initialises
    with specified parameters. Used when splitting the path over:
    - Global
    - Sliding
    - Expanding
    - Dyadic
    window types.
    Code based on window code written by Patrick Kidger.

    Parameters
    ----------
    window_name: str, String from ['global', 'sliding', 'expanding', 'dyadic']
        used to access the window method.
    window_depth: int, The depth of the dyadic window. (Active only if
        `window_name == 'dyadic']`).
    window_length: int, The length of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding']`).
    window_step: int, The step of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding']`).

    Returns
    -------
    list:
        A list of lists where the inner lists are lists of tuples that
        denote the start and end indexes of each window.
    """
    # Setup all available windows here
    length_step = {"length": window_length, "step": window_step}
    window_dict = {
        "global": (_Global, {}),
        "sliding": (_Sliding, length_step),
        "expanding": (_Expanding, length_step),
        "dyadic": (_Dyadic, {"depth": window_depth}),
    }

    if window_name not in window_dict.keys():
        raise ValueError(
            "Window name must be one of: {}. Got: {}.".format(
                window_dict.keys(), window_name
            )
        )

    window_cls, window_kwargs = window_dict[window_name]
    return window_cls(**window_kwargs)


class _Window:
    """Abstract base class for windows.

    Each subclass must implement a __call__ method that returns a list of lists
    of 2-tuples. Each 2-tuple specifies the start and end of each window.

    These windows are grouped into a list that will (usually) cover the full
    time series. These lists are grouped into another list for situations
    where we consider windows of multiple scales.
    """

    def num_windows(self, length):
        """Get the total number of windows in the set.

        Parameters
        ----------
        length: int, The length of the input path.

        Returns
        -------
        int: The number of windows.
        """
        return sum([len(w) for w in self(length)])


class _Global(_Window):
    """A single window over the full data."""

    def __call__(self, length=None):
        return [[_Pair(None, None)]]


class _ExpandingSliding(_Window):
    def __init__(self, initial_length, start_step, end_step):
        """Initialize the class.

        Parameters
        ----------
        initial_length: int, Initial length of the input window.
        start_step: int, Initial step size.
        end_step: int, Final step size.
        """
        super().__init__()
        self.initial_length = initial_length
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, length):
        def _call():
            start = 0
            end = self.initial_length
            while end <= length:
                yield _Pair(start, end)
                start += self.start_step
                end += self.end_step

        windows = list(_call())
        if len(windows) == 0:
            raise ValueError(f"Length {length} too short for given window parameters.")
        return [windows]


class _Sliding(_ExpandingSliding):
    """Sliding expanding window.

    A window starting at zero and going to some point that increases
    between windows.
    """

    def __init__(self, length, step):
        """Initialize the class.

        Parameters
        ----------
        length: int, The length of the window.
        step: int, The sliding step size.
        """
        super().__init__(initial_length=length, start_step=step, end_step=step)


class _Expanding(_ExpandingSliding):
    """A window of fixed length, slid along the dataset."""

    def __init__(self, length, step):
        """Initialize the class.

        Parameters
        ----------
        length: int, The length of each window.
        step: int, The step size.
        """
        super().__init__(initial_length=length, start_step=0, end_step=step)


class _Dyadic(_Window):
    """Windows generated 'dyadically'.

    These are successive windows of increasing fineness. The windows are as
    follows:
        Depth 1: The global window over the full data.
        Depth 2: The windows of the first and second halves of the dataset.
        Depth 3: The dataset is split into quarters, and we take the windows of
            each quarter.
        ...
        Depth n: For a dataset of length L, we generate windows
            [0:L/(2^n), L/(2^n):(2L)/(2^n), ..., (2^(n-1))L/2^n:L].
    Each depth also contains all previous depths.

    Note: Ensure the depth, n, is chosen such that L/(2^n) >= 1, else it will
        be too high for the dataset.

    Parameters
    ----------
    depth: int, The depth of the dyadic window, explained in the class
        description.
    """

    def __init__(self, depth):
        super().__init__()
        self.depth = depth

    def __call__(self, length):
        max_depth = int(np.floor(np.log2(length)))
        if self.depth > max_depth:
            raise ValueError(
                "Chosen dyadic depth is too high for the data length. "
                "We require depth <= {} for length {}. "
                "Depth given is: {}.".format(max_depth, length, self.depth)
            )
        return self.call(float(length))

    def call(self, length, _offset=0.0, _depth=0, _out=None):
        if _out is None:
            _out = [[] for _ in range(self.depth + 1)]
        _out[_depth].append(_Pair(int(_offset), int(_offset + length)))

        if _depth < self.depth:
            left = _Dyadic(self.depth)
            right = _Dyadic(self.depth)
            half_length = length / 2
            # The order of left then right is important, so that they add their
            # entries to _out in the correct order.
            left.call(half_length, _offset, _depth + 1, _out)
            right.call(half_length, _offset + half_length, _depth + 1, _out)

        return _out


def _rescale_path(path, depth):
    """Rescale input path by depth! ** (1 / depth)."""
    coeff = math.factorial(depth) ** (1 / depth)
    return coeff * path


def _rescale_signature(signature, channels, depth):
    """Rescales the output signature by multiplying the depth-d term by d!."""
    import esig

    # Needed for weird esig fails
    if depth == 1:
        sigdim = channels
    elif channels == 1:
        sigdim = depth
    else:
        sigdim = esig.sigdim(channels, depth) - 1
    # Verify shape
    if sigdim != signature.shape[-1]:
        raise ValueError(
            "A path with {} channels to depth {} should yield a "
            "signature with {} features. Input signature has {} "
            "features which is inconsistent.".format(
                channels, depth, sigdim, signature.shape[-1]
            )
        )

    end = 0
    term_length = 1
    val = 1
    terms = []
    for d in range(1, depth + 1):
        start = end
        term_length *= channels
        end = start + term_length

        val *= d

        terms.append(signature[..., start:end] * val)

    return np.concatenate(terms, axis=-1)
