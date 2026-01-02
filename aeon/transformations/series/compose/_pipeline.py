"""Pipeline with series transformers."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["SeriesTransformerPipeline"]

from aeon.base._estimators.compose.series_pipeline import BaseSeriesPipeline
from aeon.transformations.series import BaseSeriesTransformer
from aeon.transformations.series.compose import SeriesId
from aeon.utils.data_types import VALID_SERIES_INNER_TYPES


class SeriesTransformerPipeline(BaseSeriesPipeline, BaseSeriesTransformer):
    """Pipeline of series transformers.

    The `SeriesTransformerPipeline` compositor chains transformers.
    The pipeline is constructed with a list of aeon transformers,
        i.e., estimators following the BaseTransformer interface.
    The transformer list can be unnamed - a simple list of transformers -
        or string named - a list of pairs of string, estimator.

    For a list of transformers `trafo1`, `trafo2`, ..., `trafoN`,
        the pipeline behaves as follows:
    `fit(X, y)` - changes state by running `trafo1.fit_transform` on `X`,
        them `trafo2.fit_transform` on the output of `trafo1.fit_transform`, etc
        sequentially, with `trafo[i]` receiving the output of `trafo[i-1]`,
        and then running `trafo[N].fit` with `X` being the output of `trafo[N-1]`,
        and `y` identical with the input to `self.fit`
    `transform(X, y)` - result is of executing `trafo1.transform`, `trafo2.transform`,
        etc with `trafo[i].transform` input = output of `trafo[i-1].transform`,
        then running `trafo[N].transform` on the output of `trafo[N-1].transform`,
        and returning the output.

    Parameters
    ----------
    transformers : aeon or sklearn transformer or list of transformers
        A transform or list of transformers.
        List of tuples (str, transformer) of transformers can also be passed, where
        the str is used to name the transformer.
        The objecst are cloned prior, as such the state of the input will not be
        modified by fitting the pipeline.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of transformers
        Clones of transformers which are fitted in the pipeline.
        Will always be in (str, estimator) format, even if transformers input is a
        singular transform or list of transformers.

    Examples
    --------
    >>> from aeon.transformations.series import LogTransformer
    >>> from aeon.transformations.series.smoothing import MovingAverage
    >>> from aeon.datasets import load_airline
    >>> from aeon.transformations.series.compose import SeriesTransformerPipeline
    >>> X = load_airline()
    >>> pipeline = SeriesTransformerPipeline(
    ...     [LogTransformer(), MovingAverage()]
    ... )
    >>> pipeline.fit(X)
    SeriesTransformerPipeline(...)
    >>> Xt = pipeline.transform(X)
    """

    _tags = {
        "X_inner_type": VALID_SERIES_INNER_TYPES,
    }

    def __init__(self, transformers):
        if not isinstance(transformers, list):
            transformers = [SeriesId(), transformers]
        elif len(transformers) < 2:
            transformers = [SeriesId(), transformers[0]]

        super().__init__(transformers=transformers, _estimator=None)

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
        from aeon.transformations.series import LogTransformer
        from aeon.transformations.series.smoothing import MovingAverage

        return {
            "transformers": [
                LogTransformer(),
                MovingAverage(),
            ]
        }
