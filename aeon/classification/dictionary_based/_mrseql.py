"""Multiple Representations Sequence Learning (MrSEQL) Classifier."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["MrSEQLClassifier"]

from typing import Union

import numpy as np
import pandas as pd

from aeon.classification import BaseClassifier


def _from_numpy3d_to_nested_dataframe(X):
    """Convert numpy3D collection to a pd.DataFrame where each cell is a series."""
    n_cases, n_channels, n_timepoints = X.shape
    array_type = X.dtype
    container = pd.Series
    column_names = [f"channel_{i}" for i in range(n_channels)]
    column_list = []
    for j, column in enumerate(column_names):
        nested_column = (
            pd.DataFrame(X[:, j, :])
            .apply(lambda x: [container(x, dtype=array_type)], axis=1)
            .str[0]
            .rename(column)
        )
        column_list.append(nested_column)
    df = pd.concat(column_list, axis=1)
    return df


class MrSEQLClassifier(BaseClassifier):
    """
    Multiple Representations Sequence Learning (MrSEQL) Classifier.

    This is a wrapper for the MrSEQLClassifier algorithm from the `mrseql` package.
    MrSEQL is not included in ``all_extras`` as it requires gcc and fftw
    (http://www.fftw.org/index.html) to be installed for Windows and some Linux OS.

    Overview: MrSEQL extends the symbolic sequence classifier (SEQL) to work with
    multiple symbolic representations of time series, using features extracted from the
    SAX and SFA transformations.

    Parameters
    ----------
    seql_mode : "clf" or "fs", default="fs".
        If "fs", trains a logistic regression model with features extracted by SEQL.
        IF "clf", builds an ensemble of SEQL models
    symrep : "sax" or "sfa", or ["sax", "sfa"], default = "sax"
        The symbolic features to extract from the time series.
    custom_config : dict, default=None
        Additional configuration for the symbolic transformations. See the original
        package for details. ``symrep`` will be ignored if used.

    References
    ----------
    .. [1] Le Nguyen, Thach, et al. "Interpretable time series classification using
        linear models and multi-resolution multi-domain symbolic representations."
        Data mining and knowledge discovery 33 (2019): 1183-1222.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import MrSEQLClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(random_state=0)
    >>> clf = MrSEQLClassifier(random_state=0) # doctest: +SKIP
    >>> clf.fit(X, y) # doctest: +SKIP
    MrSEQLClassifier(...)
    >>> clf.predict(X) # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "algorithm_type": "dictionary",
        "python_dependencies": "mrseql",
    }

    def __init__(self, seql_mode="fs", symrep=("sax"), custom_config=None) -> None:
        self.seql_mode = seql_mode
        self.symrep = symrep
        self.custom_config = custom_config

        super().__init__()

    def _fit(self, X, y):
        from mrseql import MrSEQLClassifier

        _X = _from_numpy3d_to_nested_dataframe(X)

        self.clf_ = MrSEQLClassifier(
            seql_mode=self.seql_mode,
            symrep=self.symrep,
            custom_config=self.custom_config,
        )
        self.clf_.fit(_X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        _X = _from_numpy3d_to_nested_dataframe(X)
        return self.clf_.predict(_X)

    def _predict_proba(self, X) -> np.ndarray:
        _X = _from_numpy3d_to_nested_dataframe(X)
        return self.clf_.predict_proba(_X)

    @classmethod
    def _get_test_params(
        cls, parameter_set: str = "default"
    ) -> Union[dict, list[dict]]:
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {}
