"""Class to broadcast a single series transformer over a collection."""

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.series import BaseSeriesTransformer


class BroadcastTransformer(BaseCollectionTransformer):
    """Broadcasts a single series transformer over a collection.

    Uses the single series transformer passed in the constructor over a collection of
    series. Design points to note:

    1. This class takes its capabilities from the series transformer. So, for example,
    it will only work with a collection of multivariate series if the single series
    transformer has the ``capability:multivariate`` tag set to True.
    2. If the tag `fit_is_empty` is True, it will use the same instance of the series
    transformer for each series in the collection. If `fit_is_empty` is False,
    it will clone the single series transformer for each instance and save the fitted
    version for each series.

    Parameters
    ----------
    transformer : BaseSeriesTransformer
        The single series transformer to broadcast accross the collection.
    """

    def __init__(
        self,
        transformer: BaseSeriesTransformer,
    ):
        self.transformer = transformer
        # Copy tags from transformer
        self.set_tags(**transformer.get_tags())
        super().__init__()

    def _fit(self, X, y=None):
        """Only reachable if fit_is_empty is false."""
        self.single_transformers_ = [self.transformer.clone() for _ in range(len(X))]
        for i in range(len(X)):
            self.single_transformers_[i]._fit(X[i])

    def _transform(self, X, y=None):
        """If fit is empty is true only single transform."""
        Xt = []
        if self.get_tag("fit_is_empty"):
            for i in range(len(X)):
                Xt.append(self.transformer._transform(X[i]))
        else:
            for i in range(len(X)):
                Xt.append(self.single_transformers_[i]._transform(X[i]))
        return Xt
