"""Collection transformer that contains a single series transformer."""

__author__ = ["TonyBagnall"]
__all__ = ["CollectionTransformerWrapper"]
from aeon.transformations.collection.base import BaseCollectionTransformer


class CollectionTransformerWrapper(BaseCollectionTransformer):
    """Collection transformer that contains a single series transformer.

    This class takes a single ``SeriesTransformer`` and applies it to each series
    independently through calls to ``fit_transform``.

    Cannot be used with ``y``.

    LEAVE HERE FOR DISCUSSION: It can either use the same SeriesTransformer for each
    series
    ``clone_transformer == False``, or it can clone a new ``SeriesTransformer`` for each
    series if ``clone_transformer == True``. In both cases, it makes little sense to
    separate the ``fit`` and ``transform`` steps, because even if operations are
    performed in ``fit`` it assumes the same ``X`` is passed in ``transform``. Hence
    this should only be used with transform, and calling fit directly with raise an
    error.

    Parameters
    ----------
    base_transformer : BaseTransformer
        The transformer to apply to each series independently.
    REMOVED THIS
    clone_transformer : bool, default=False
        If True, a clone of the transformer is used for each series. This makes sense if
        a different model is fitted to each series, e.g. an arima model.
        If False, the same transformer instance is used for each series. It is
        assumed that fit does not need to be called in this instance, and that
        transform alone is sufficient.

    """

    def __init__(self, base_transformer):
        self.base_transformer = base_transformer
        # Copy tags for base transformer
        super(CollectionTransformerWrapper, self).__init__()
        multivariate = base_transformer.get_tag("capability:multivariate")
        inverse = base_transformer.get_tag("capability:inverse_transform")
        fit_is_empty = base_transformer.get_tag("fit_is_empty")
        requires_y = base_transformer.get_tag("requires_y")
        tags_to_set = {
            "capability:multivariate": multivariate,
            "capability:fit_is_empty": fit_is_empty,
            "requires_y": requires_y,
            "capability:inverse_transform": inverse,
        }
        self.set_tags(**tags_to_set)

    def _transform(self, X, y=None):
        Xt = []
        for x in X:
            Xt.append(self.transformer.fit_transform(x))
        return Xt

    def _fit_transform(self, X, y=None):
        return self._transform(self, X)
