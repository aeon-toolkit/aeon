"""Class to wrap a single series transformer over a collection."""

__maintainer__ = ["baraline"]
__all__ = ["SeriesToCollectionWrapper"]

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import get_n_cases


class SeriesToCollectionWrapper(BaseCollectionTransformer):
    """Wrap a single series transformer over a collection.

    Uses the single series transformer passed in the constructor over a
    collection of series.

    Parameters
    ----------
    transformer : BaseSeriesTransformer
        The single series transformer to broadcast over the collection.

    Examples
    --------
    >>>    from aeon.transformations.series import DummySeriesTransformer
    >>> import numpy as np
    >>> X = np.np.random.rand(4, 1, 10)
    >>> transformer = DummySeriesTransformer()
    >>> X_t = transformer.fit_transform(X)
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
        self.transformer = transformer
        series_tags = transformer.get_tags()
        tags_to_add = self._tags
        for key in series_tags:
            tags_to_add.pop(key, None)
        super().__init__()
        # Setting tags before __init__() cause them to be overwriten.
        self.set_tags(**tags_to_add)

    def _fit(self, X, y=None):
        """
        Clone and fit instances of the transformer independently for each sample.

        This Function only reachable if fit_is_empty is false.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
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
            self.single_transformers_[i]._fit(X[i])

    def _transform(self, X, y=None):
        """
        Use transform function of each transformer independently for each sample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : None,
            Present for interface compatibility.

        Raises
        ------
        ValueError
            When fit_is_empty is False, a ValueError can be raised if the
            size of X is different of the size of series_transformers. This
            indicates that the input may different of the one given during
            fit. As a BaseSeriesTransformer is only fitted to a single series,
            it only makes sense to use transform with the same series in a
            wrapping context.

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
    def get_test_params(cls, parameter_set="default"):
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
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from aeon.transformations.series import DummySeriesTransformer

        return {"transformer": DummySeriesTransformer()}
