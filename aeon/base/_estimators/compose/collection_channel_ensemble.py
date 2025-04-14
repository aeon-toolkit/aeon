"""Base class for composable channel ensembles in series collection modules.

i.e. classification, regression and clustering.
"""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["BaseCollectionChannelEnsemble"]

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from aeon.base import (
    BaseAeonEstimator,
    BaseCollectionEstimator,
    ComposableEstimatorMixin,
)
from aeon.base._base import _clone_estimator


class BaseCollectionChannelEnsemble(ComposableEstimatorMixin, BaseCollectionEstimator):
    """Applies estimators to channels of an array.

    Parameters
    ----------
    _ensemble : list of aeon and/or sklearn estimators or list of tuples
        Estimators to be used in the ensemble.
        A list of tuples (str, estimator) can also be passed, where the str is used to
        name the estimator.
        The objects are cloned prior. As such, the state of the input will not be
        modified by fitting the ensemble.
    channels : list of int, array-like of int, slice, "all", "all-split" or callable
        Channel(s) to be used with the estimator. Must be the same length as
        ``_ensemble``.
        If "all", all channels are used for the estimator. "all-split" will create a
        separate estimator for each channel.
        int, array-like of int and slice are used as indices to select channels. If a
        callable is passed, the input data should return the channel indices to be used.
    remainder : BaseEstimator or None, default=None
        By default, only the specified channels in ``channels`` are
        used and combined in the output, and the non-specified
        channels are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support ``fit`` and ``predict``.
    random_state : int, RandomState instance or None, default=None
        Random state used to fit the estimators. If None, no random state is set for
        ensemble members (but they may still be seeded prior to input).
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
    _ensemble_input_name : str, default="estimators"
        Name of the input parameter for the ensemble of estimators.

    Attributes
    ----------
    ensemble_ : list of tuples (str, estimator) of estimators
        Clones of estimators in _ensemble which are fitted in the ensemble.
        Will always be in (str, estimator) format regardless of _ensemble input.
    channels_ : list
        The channel indices for each estimator in ``ensemble_``.

    See Also
    --------
    ClassifierChannelEnsemble : A channel ensemble for classification tasks.
    """

    # Attribute name containing an iterable of processed (str, estimator) tuples
    # with unfitted estimators and unique names. Used in get_params and set_params
    _estimators_attr = "_ensemble"
    # Attribute name containing an iterable of fitted (str, estimator) tuples.
    # Used in get_fitted_params
    _fitted_estimators_attr = "ensemble_"

    @abstractmethod
    def __init__(
        self,
        _ensemble,
        channels,
        remainder=None,
        random_state=None,
        _ensemble_input_name="estimators",
    ):
        self._ensemble = _ensemble
        self.channels = channels
        self.remainder = remainder
        self.random_state = random_state
        self._ensemble_input_name = _ensemble_input_name

        self._check_estimators(
            self._ensemble,
            attr_name=_ensemble_input_name,
            class_type=BaseEstimator,
            invalid_names=["Remainder"],
        )
        self._ensemble = self._convert_estimators(
            self._ensemble, clone_estimators=False
        )

        super().__init__()

        # can handle missing values if all estimators can
        missing = all(
            [
                (
                    e[1].get_tag(
                        "capability:missing_values",
                        raise_error=False,
                        tag_value_default=False,
                    )
                    if isinstance(e[1], BaseAeonEstimator)
                    else False
                )
                for e in self._ensemble
            ]
        )
        remainder_missing = remainder is None or (
            isinstance(remainder, BaseAeonEstimator)
            and remainder.get_tag(
                "capability:missing_values", raise_error=False, tag_value_default=False
            )
        )

        # can handle unequal length if all estimators can
        unequal = all(
            [
                (
                    e[1].get_tag(
                        "capability:unequal_length",
                        raise_error=False,
                        tag_value_default=False,
                    )
                    if isinstance(e[1], BaseAeonEstimator)
                    else False
                )
                for e in self._ensemble
            ]
        )
        remainder_unequal = remainder is None or (
            isinstance(remainder, BaseAeonEstimator)
            and remainder.get_tag(
                "capability:unequal_length", raise_error=False, tag_value_default=False
            )
        )

        tags_to_set = {
            "capability:missing_values": missing and remainder_missing,
            "capability:unequal_length": unequal and remainder_unequal,
        }
        self.set_tags(**tags_to_set)

    def _fit(self, X, y):
        n_channels = X[0].shape[0]
        rng = check_random_state(self.random_state)

        # clone estimators
        self.ensemble_ = [
            (
                step[0],
                _clone_estimator(
                    step[1], random_state=rng.randint(np.iinfo(np.int32).max)
                ),
            )
            for step in self._ensemble
        ]

        # verify channels list
        if not isinstance(self.channels, list):
            raise ValueError("channels must be a list of valid inputs, see docstring.")
        if len(self.channels) != len(self._ensemble):
            raise ValueError(
                "The number of channels must be the same as the number of estimators."
            )

        # process channels options
        msg = (
            "Selected estimator channels must be a int, list/tuple of ints, "
            "slice, 'all' or 'all-split' (or a callable resulting in one of these)."
        )
        splits = []
        self.channels_ = []
        for i, channel in enumerate(self.channels):
            if callable(channel):
                channel = channel(X)

            if channel == "all":
                channel = list(range(n_channels))
            elif channel == "all-split":
                splits.append(i)
            elif isinstance(channel, slice):
                if not isinstance(channel.start, (int, type(None))) or not isinstance(
                    channel.stop, (int, type(None))
                ):
                    raise ValueError(msg)
            elif isinstance(channel, (list, tuple)):
                if not all(isinstance(x, int) for x in channel):
                    raise ValueError(msg)
            elif not isinstance(channel, int):
                raise ValueError(msg)

            self.channels_.append(channel)

        # if any channels are all-split, create a separate estimator for each channel
        for i in splits:
            self.ensemble_[i] = (self.ensemble_[i][0] + "-0", self.ensemble_[i][1])
            self.channels_[i] = 0
            for n in range(1, n_channels):
                self.ensemble_.append(
                    (
                        self.ensemble_[i][0] + "-" + str(n),
                        _clone_estimator(
                            self.ensemble_[i][1],
                            random_state=rng.randint(np.iinfo(np.int32).max),
                        ),
                    )
                )
                self.channels_.append(n)

        # process remainder if not None
        if self.remainder is not None:
            current_channels = []
            all_channels = np.arange(n_channels)
            for channels in self._channels:
                if isinstance(channels, int):
                    channels = [channels]
                current_channels.extend(all_channels[channels])
            remaining_idx = sorted(list(set(all_channels) - set(current_channels)))

            if remaining_idx:
                self.ensemble_.append(
                    (
                        "Remainder",
                        _clone_estimator(
                            self.remainder,
                            random_state=rng.randint(np.iinfo(np.int32).max),
                        ),
                    )
                )
                self.channels_.append(remaining_idx)

        # fit estimators
        for i, (_, estimator) in enumerate(self.ensemble_):
            estimator.fit(self._get_channel(X, self.channels_[i]), y)

        return self

    @staticmethod
    def _get_channel(X, key):
        """Get time series channel(s) from input data X."""
        if isinstance(X, np.ndarray):
            return X[:, key]
        else:
            li = [x[key] for x in X]
            if li[0].ndim == 1:
                li = [x.reshape(1, -1) for x in li]
            return li
