"""Class to broadcast a single series transformer over a collection."""

__maintainer__ = ["baraline"]
__all__ = ["BroadcastTransformer"]

import numpy as np
from joblib import Parallel, delayed

from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.series.base import BaseSeriesTransformer
from aeon.utils.validation import check_n_jobs


def _joblib_container_fit(
    transformer: BaseSeriesTransformer, X, y
) -> BaseSeriesTransformer:
    return transformer.fit(X, y=y)


def _joblib_container_transform(transformer: BaseSeriesTransformer, X, y):
    return transformer.transform(X, y=y)


def _joblib_container_inverse_transform(transformer: BaseSeriesTransformer, X, y):
    return transformer.inverse_transform(X, y=y)


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
    n_jobs : int, optional
        Number of jobs to used. This will be used call the methods of the
        SeriesTransformer in parallel. The default is 1.
    joblib_backend : str, optional
        Joblib backend to use. The default is "loky".

    Examples
    --------
    >>> from aeon.transformations.collection import BroadcastTransformer
    >>> from aeon.transformations.series import DummySeriesTransformer
    >>> from aeon.datasets import load_unit_test
    >>> X, y = load_unit_test()
    >>> transformer = BroadcastTransformer(DummySeriesTransformer())
    >>> X_t = transformer.fit_transform(X)
    """

    _tags_to_inherit = [
        "capability:unequal_length",
        "capability:missing_values",
        "univariate-only",
        "capability:multivariate",
        "capability:inverse_transform",
        "requires_y",
        "transform_labels",
        "fit_is_empty",
        "X-y-must-have-same-index",
        "skip-inverse-transform",
    ]

    def __init__(
        self,
        transformer: BaseSeriesTransformer,
        n_jobs: int = 1,
        joblib_backend: str = "loky",
    ) -> None:
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend
        super().__init__()
        # Setting tags before __init__() cause them to be overwriten.
        _tags = {key: transformer.get_tags()[key] for key in self._tags_to_inherit}
        if _tags["fit_is_empty"]:
            transformer._is_fitted = True
        self.set_tags(**_tags)

    def _check_n_jobs_broadcast(self, n_cases: int) -> (int, int):
        """
        Check the n_jobs parameters of the broadcaster and the transform.

        It will compute a balanced number of jobs by dividing the n_jobs parameter
        defined in the transformer by the n_jobs parameter defined in the broadcaster.
        If the transformer does not have a n_jobs parameter, it will only use the
        n_jobs of the broadcaster.


        Parameters
        ----------
        n_cases : int
            Number of samples to transform. Used to limit the number of jobs of the
            broadcaster, which don't need to be higher than the number of instances,
            as it will loop on them.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        int
            Number of jobs for the broadcaster.
        int
            Number of jobs for the transformer.

        """
        true_n_jobs = check_n_jobs(self.n_jobs)
        if true_n_jobs > 1:
            if hasattr(self.transformer, "n_jobs"):
                transformer_n_jobs = check_n_jobs(self.transformer.n_jobs)
            else:
                transformer_n_jobs = 1

            joblib_n_jobs = true_n_jobs // transformer_n_jobs
            if joblib_n_jobs > 0:
                return min(n_cases, joblib_n_jobs), transformer_n_jobs
            else:
                raise ValueError(
                    f"Got {joblib_n_jobs} jobs for parallel broadcasting of "
                    f"{self.transformer.__class__.__name__}, this value should "
                    "be strictly superior to zero."
                )
        else:
            return 1, 1

    def _fit(self, X, y=None):
        """
        Clone and fit instances of the transformer independently for each sample.

        This Function only reachable if fit_is_empty is false.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : 1D np.ndarray of shape = (n_cases), optional
            Class of the samples. The default is None, which means this parameter
            is ignored.

        Returns
        -------
        None.

        """
        n_samples = len(X)
        if y is None:
            y = [None] * n_samples

        n_jobs_joblib, _ = self._check_n_jobs_broadcast(n_samples)
        self.series_transformers = Parallel(
            n_jobs=n_jobs_joblib, backend=self.joblib_backend
        )(
            delayed(_joblib_container_fit)(self.transformer.clone(), X[i], y[i])
            for i in range(len(X))
        )

    def _transform(self, X, y=None):
        """
        Call the transform function of each transformer independently for each sample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : 1D np.ndarray of shape = (n_cases), optional
            Class of the samples. The default is None, which means this parameter
            is ignored.

        Returns
        -------
        Xt : np.ndarray
            The transformed collection of time series.

        """
        n_samples = len(X)
        if y is None:
            y = [None] * n_samples

        n_jobs_joblib, _ = self._check_n_jobs_broadcast(n_samples)
        if self.get_tag("fit_is_empty"):
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_transform)(self.transformer, X[i], y[i])
                for i in range(len(X))
            )
        else:
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_transform)(
                    self.series_transformers[i], X[i], y[i]
                )
                for i in range(len(X))
            )
        return np.asarray(Xt)

    def _inverse_transform(self, X, y=None):
        """
        Call the inverse_transform function of each transformer for each sample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : 1D np.ndarray of shape = (n_cases), optional
            Class of the samples. The default is None, which means this parameter
            is ignored.

        Returns
        -------
        Xt : np.ndarray
            The transformed collection of time series.

        """
        n_samples = len(X)
        if y is None:
            y = [None] * n_samples

        n_jobs_joblib, _ = self._check_n_jobs_broadcast(n_samples)
        if self.get_tag("fit_is_empty"):
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_inverse_transform)(
                    self.transformer, X[i], y[i]
                )
                for i in range(len(X))
            )
        else:
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_inverse_transform)(
                    self.series_transformers[i], X[i], y[i]
                )
                for i in range(len(X))
            )
        return np.asarray(Xt)

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
