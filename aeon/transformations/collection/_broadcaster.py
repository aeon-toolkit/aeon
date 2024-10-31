"""Class to wrap a single series transformer over a collection."""

__maintainer__ = ["baraline", "TonyBagnall"]
__all__ = ["SeriesToCollectionBroadcaster"]

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import get_n_cases


class SeriesToCollectionBroadcaster(BaseCollectionTransformer):
    """Broadcast a ``BaseSeriesTransformer`` over a collection of time series.

    Uses the ``BaseSeriesTransformer`` passed in the constructor.  If the
    BaseSeriesTransformer has no fit function (tag ``"fit_is_empty": True"``) we
    use a single instance of the transformer on every series. If the series
    transformer has a fit function, we clone a transformer for each series.

    Parameters
    ----------
    transformer : BaseSeriesTransformer
        The single series transformer to broadcast over the collection.

    Examples
    --------
    >>> from aeon.transformations.series._boxcox import BoxCoxTransformer
    >>> import numpy as np
    >>> X = np.random.rand(4, 1, 10)
    >>> transformer = BoxCoxTransformer()
    >>> broadcaster = SeriesToCollectionBroadcaster(transformer)
    >>> X_t = broadcaster.fit_transform(X)
    """

    # These tags are not set from the series transformer.
    _tags = {
        "input_data_type": "Collection",
        "output_data_type": "Collection",
        "capability:unequal_length": True,
        "X_inner_type": ["numpy3D", "np-list"],
    }

    def __init__(
        self,
        transformer: BaseSeriesTransformer,
    ) -> None:
        # Setting tags before __init__() causes them to be overwritten. Hence we make
        # a copy before init from the series transformer, then copy the tags of the
        # BaseSeriesTransformer to this BaseCollectionTransformer
        self.transformer = transformer
        tags_to_keep = SeriesToCollectionBroadcaster._tags
        tags_to_add = transformer.get_tags()
        for key in tags_to_keep:
            tags_to_add.pop(key, None)
        super().__init__()
        self.set_tags(**tags_to_add)

    def _fit(self, X, y=None):
        """
        Clone and fit instances of the transformer independently for each sample.

        This Function only reachable if fit_is_empty is false, hence we always clone
        the transformer in this case.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints), or list of
        2D np.ndarray
            The collection of time series to transform.
        y : None,
            Present for interface compatibility.

        Returns
        -------
        None.

        """
        n_cases = get_n_cases(X)
        self.single_transformers_ = [self.transformer.clone() for _ in range(n_cases)]
        for i in range(n_cases):
            self.single_transformers_[i].fit(X[i])

    def _transform(self, X, y=None):
        """
        Use transform function of each transformer independently for each sample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : None,
            Present for interface compatibility.

        Returns
        -------
        Xt : np.ndarray or list
            The transformed collection of time series, either a 3D numpy or a
            list of 2D numpy.

        """
        n_cases = get_n_cases(X)
        """If fit is empty is true only single transform is used."""
        Xt = []
        if self.get_tag("fit_is_empty"):
            for i in range(n_cases):
                Xt.append(self.transformer._transform(X[i]))
        else:
            for i in range(n_cases):
                Xt.append(self.single_transformers_[i]._transform(X[i]))
        # Need to make it a valid collection
        for i in range(n_cases):
            if isinstance(Xt[i], np.ndarray) and Xt[i].ndim == 1:
                Xt[i] = Xt[i].reshape(1, -1)
        return Xt

    def _inverse_transform(self, X, y=None):
        """
        Call the inverse_transform function of each transformer for each sample.

        Only reachable if the tag "capability:inverse_transform" of the base
        transformer is True.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : None,
            Present for interface compatibility.

        Returns
        -------
        Xt : np.ndarray or list
            The transformed collection of time series, either a 3D numpy or a
            list of 2D numpy.
        """
        n_cases = len(X)
        Xt = []
        if self.get_tag("fit_is_empty"):
            for i in range(n_cases):
                Xt.append(self.transformer._inverse_transform(X[i]))
        else:
            for i in range(n_cases):
                Xt.append(self.single_transformers_[i]._inverse_transform(X[i]))
        return Xt

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
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        from aeon.testing.mock_estimators._mock_series_transformers import (
            MockUnivariateSeriesTransformer,
        )

        return {"transformer": MockUnivariateSeriesTransformer()}
