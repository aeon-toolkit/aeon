"""Class to wrap a single series transformer over a collection."""

__maintainer__ = ["baraline"]
__all__ = ["SeriesToCollectionWrapper"]

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


class SeriesToCollectionWrapper(BaseCollectionTransformer):
    """Wrap a single series transformer over a collection.

    Uses the single series transformer passed in the constructor over a
    collection of series.

    Parameters
    ----------
    transformer : BaseSeriesTransformer
        The single series transformer to warp accross the collection.
    n_jobs : int, optional
        Number of jobs to used. This will be used call the methods of the
        SeriesTransformer in parallel. The default is 1.
    joblib_backend : str, optional
        Joblib backend to use. The default is "loky".

    Examples
    --------
    >>> from aeon.transformations.collection import SeriesToCollectionWrapper
    >>> from aeon.transformations.series import DummySeriesTransformer
    >>> from aeon.datasets import load_unit_test
    >>> X, y = load_unit_test()
    >>> transformer = SeriesToCollectionWrapper(DummySeriesTransformer())
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
        n_jobs: int = None,
        joblib_backend: str = "threading",
    ) -> None:
        self.transformer = transformer
        self.n_jobs = n_jobs
        self.joblib_backend = joblib_backend
        super().__init__()
        # Setting tags before __init__() cause them to be overwriten.
        _tags = {key: transformer.get_tags()[key] for key in self._tags_to_inherit}
        self.set_tags(**_tags)
        if _tags["fit_is_empty"]:
            transformer._is_fitted = True

    def _check_n_jobs_wrapper(self, n_cases: int) -> (int, int):
        """
        Check the n_jobs parameters of the wrapper and the transform.

        It will compute a balanced number of jobs by dividing the n_jobs parameter
        defined in the transformer by the n_jobs parameter defined in the wrapper.
        If the transformer does not have a n_jobs parameter, it will only use the
        n_jobs of the wrapper.


        Parameters
        ----------
        n_cases : int
            Number of samples to transform. Used to limit the number of jobs of the
            wrapper, which don't need to be higher than the number of instances,
            as it will loop on them.

        Raises
        ------
        ValueError
            Raise a ValueError when the number of parallel jobs affected to
            the wrapper is not stricly superior to zero. This number is
            computed based on the n_jobs parameter given in the initialization
            of both the wrapper and the transformer. The error
            will be raised when transformer_n_jobs > CPU count or when
            wrapper_n_jobs // transformer_n_jobs <= 0.


        Returns
        -------
        int
            Number of jobs for the wrapper.
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
                    f"Got {joblib_n_jobs} jobs for parallel wrapper of "
                    f"{self.transformer.__class__.__name__}, this value should "
                    "be strictly superior to zero."
                )
        else:
            return 1, 1

    def _set_n_jobs_transformer(self, n_jobs: int):
        """
        Set the number of jobs for the transformer objects.

        We don't use set_params to avoid calling reset(). If fit is not empty,
        series_transformer are cloned from the transformer object, so setting
        n_jobs for series_transformer is only needed after fit has been called.

        Parameters
        ----------
        n_jobs : int
            Number of jobs to affect to transformer objects

        Returns
        -------
        None.

        """
        if hasattr(self.transformer, "n_jobs"):
            self.transformer.n_jobs = n_jobs

            if self.is_fitted and not self.get_tag("fit_is_empty"):
                for i in range(len(self.series_transformers)):
                    self.series_transformers[i].n_jobs = n_jobs

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

        n_jobs_joblib, n_jobs_transformer = self._check_n_jobs_wrapper(n_samples)
        self._set_n_jobs_transformer(n_jobs_transformer)

        self.series_transformers = Parallel(
            n_jobs=n_jobs_joblib, backend=self.joblib_backend
        )(
            delayed(_joblib_container_fit)(self.transformer.clone(), X[i], y[i])
            for i in range(len(X))
        )

    def _transform(self, X, y=None):
        """
        Use transform function of each transformer independently for each sample.

        Parameters
        ----------
        X : 3D np.ndarray of shape = (n_cases, n_channels, n_timepoints)
            The collection of time series to transform.
        y : 1D np.ndarray of shape = (n_cases), optional
            Class of the samples. The default is None, which means this parameter
            is ignored.

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
        Xt : np.ndarray
            The transformed collection of time series.

        """
        n_samples = len(X)
        if y is None:
            y = [None] * n_samples

        n_jobs_joblib, n_jobs_transformer = self._check_n_jobs_wrapper(n_samples)
        self._set_n_jobs_transformer(n_jobs_transformer)

        if self.get_tag("fit_is_empty"):
            # Not Cloning transformer for joblib parallel with threading
            # might cause some non-efficient parallelism as we call from
            # the same object. But cloning remove deep params such as
            # _is_fitted. Processes/Loky would copy it by default.
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_transform)(self.transformer, X[i], y[i])
                for i in range(len(X))
            )
        else:
            if len(X) != len(self.series_transformers):
                raise ValueError(
                    f"The number of sample ({len(X)}) is different from the "
                    f"number of fitted transformers "
                    f"({len(self.series_transformers)}). If the wrapped "
                    "transformer needs to be fitted, you cannot call "
                    "transform with a different collection of time"
                    "series without re-fitting it first."
                )

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
        Xt : np.ndarray
            The transformed collection of time series.

        """
        n_samples = len(X)
        if y is None:
            y = [None] * n_samples

        n_jobs_joblib, n_jobs_transformer = self._check_n_jobs_wrapper(n_samples)
        self._set_n_jobs_transformer(n_jobs_transformer)

        if self.get_tag("fit_is_empty"):
            # Not Cloning transformer for joblib parallel with threading
            # might cause some non-efficient parallelism as we call from
            # the same object. But cloning remove deep params such as
            # _is_fitted. Processes/Loky would copy it by default.
            Xt = Parallel(n_jobs=n_jobs_joblib, backend=self.joblib_backend)(
                delayed(_joblib_container_inverse_transform)(
                    self.transformer, X[i], y[i]
                )
                for i in range(len(X))
            )
        else:
            if len(X) != len(self.series_transformers):
                raise ValueError(
                    f"The number of sample ({len(X)}) is different from the "
                    f"number of fitted transformers "
                    f"({len(self.series_transformers)}). If the wrapped "
                    "transformer needs to be fitted, you cannot call "
                    "inverse_transform with a different collection of time"
                    "series without re-fitting it first."
                )

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
