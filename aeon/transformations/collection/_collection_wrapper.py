"""A wrapper to treat a BaseCollectionTransformer as a BaseTransformer."""

__maintainer__ = []
__all__ = ["CollectionToSeriesWrapper"]

from aeon.base._base import _clone_estimator
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection.base import BaseCollectionTransformer


class CollectionToSeriesWrapper(BaseTransformer):
    """A wrapper to treat a BaseCollectionTransformer as a BaseTransformer.

    This will enable vectorisation, allowing single series datatypes to be used among
    other conversions contained in the BaseTransformer base functionality.
    Transformer output will be converted using the ruleset defined for BaseTransformers
    unless _output_convert is changed from "auto".
    2D numpy array input will be treated as a single multivariate series, instead of
    multiple univariate ones.

    Parameters
    ----------
    transformer : BaseCollectionTransformer
        The transformer to wrap. While a regular BaseTransformer can be used as an input
        here, it is pointless to do so.

    Attributes
    ----------
    transformer_ : BaseCollectionTransformer
        The cloned transformer input. Used to fit and transform.

    Examples
    --------
    >>> from aeon.transformations.collection import CollectionToSeriesWrapper
    >>> from aeon.transformations.collection.feature_based import Catch22
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> wrap = CollectionToSeriesWrapper(Catch22())
    >>> wrap.fit_transform(y)
               0           1     2         3   ...        18        19        20    21
    0  155.800003  181.700012  49.0  0.541667  ...  0.282051  0.769231  0.166667  11.0
    <BLANKLINE>
    [1 rows x 22 columns]
    """

    _tags = {
        "X_inner_type": BaseCollectionTransformer.ALLOWED_INPUT_TYPES,
    }

    def __init__(
        self,
        transformer,
        _output_convert="auto",
    ):
        self.transformer = transformer
        self.transformer_ = _clone_estimator(self.transformer)

        super().__init__(_output_convert=_output_convert)
        self.clone_tags(transformer)

    def _fit_transform(self, X, y=None):
        return self.transformer_.fit_transform(X, y)

    def _fit(self, X, y=None):
        return self.transformer_.fit(X, y)

    def _transform(self, X, y=None):
        return self.transformer_.transform(X, y)

    def _inverse_transform(self, X, y=None):
        return self.transformer_.inverse_transform(X, y)

    def _update(self, X, y=None):
        return self.transformer_.update(X, y)

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
        from aeon.transformations.collection.convolution_based import Rocket

        return {"transformer": Rocket(num_kernels=50)}
