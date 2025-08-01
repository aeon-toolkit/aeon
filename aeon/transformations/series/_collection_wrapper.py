"""Class to wrap a collection transformer for single series."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["CollectionToSeriesWrapper"]


from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.series.base import BaseSeriesTransformer


class CollectionToSeriesWrapper(BaseSeriesTransformer):
    """Wrap a ``BaseCollectionTransformer`` to run on single series datatypes.

    Parameters
    ----------
    transformer : BaseCollectionTransformer
        The collection transformer to wrap.

    Examples
    --------
    >>> from aeon.transformations.collection.unequal_length import Resizer
    >>> import numpy as np
    >>> X = np.random.rand(1, 10)
    >>> transformer = Resizer(length=5)
    >>> wrapper = CollectionToSeriesWrapper(transformer)
    >>> X_t = wrapper.fit_transform(X)
    """

    # These tags are not set from the collection transformer.
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "capability:inverse_transform": True,
        "X_inner_type": "np.ndarray",
    }

    def __init__(
        self,
        transformer: BaseCollectionTransformer,
    ) -> None:
        self.transformer = transformer

        super().__init__(axis=1)

        # Setting tags before __init__() causes them to be overwritten.
        tags_to_keep = CollectionToSeriesWrapper._tags
        tags_to_add = transformer.get_tags()
        for key in tags_to_keep:
            tags_to_add.pop(key, None)
        for key in ["capability:unequal_length", "removes_unequal_length"]:
            tags_to_add.pop(key, None)
        self.set_tags(**tags_to_add)

    def _fit(self, X, y=None):
        X = X.reshape(1, X.shape[0], X.shape[1])
        self.collection_transformer_ = self.transformer.clone()
        self.collection_transformer_.fit(X, y)

    def _transform(self, X, y=None):
        X = X.reshape(1, X.shape[0], X.shape[1])

        t = self.transformer
        if not self.get_tag("fit_is_empty"):
            t = self.collection_transformer_

        return t.transform(X, y)

    def _fit_transform(self, X, y=None):
        X = X.reshape(1, X.shape[0], X.shape[1])
        self.collection_transformer_ = self.transformer.clone()
        return self.collection_transformer_.fit_transform(X, y)

    def _inverse_transform(self, X, y=None):
        X = X.reshape(1, X.shape[0], X.shape[1])

        t = self.transformer
        if not self.get_tag("fit_is_empty"):
            t = self.collection_transformer_

        return t.inverse_transform(X, y)

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
        from aeon.testing.mock_estimators._mock_collection_transformers import (
            MockCollectionTransformer,
        )

        return {"transformer": MockCollectionTransformer()}
