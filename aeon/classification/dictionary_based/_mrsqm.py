"""Multiple Representations Sequence Miner (MrSQM) Classifier."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]
__all__ = ["MrSQMClassifier"]

from typing import Union

import numpy as np

from aeon.classification import BaseClassifier


class MrSQMClassifier(BaseClassifier):
    """
    Multiple Representations Sequence Miner (MrSQM) classifier.

    This is a wrapper for the MrSQMClassifier algorithm from the `mrsqm` package.
    MrSQM is not included in ``all_extras`` as it requires gcc and fftw
    (http://www.fftw.org/index.html) to be installed for Windows and some Linux OS.

    Overview: MrSQM is a time series classifier utilising symbolic
    representations of time series. MrSQM implements four different feature selection
    strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic
    representations of time series data.

    Parameters
    ----------
    strat : str, default="RS"
        Feature selection strategy. One of 'R','S','SR', or 'RS. R and S are
        single-stage filters while RS and SR are two-stage filters.
    features_per_rep : int, default=500
        The (maximum) number of features selected per representation.
    selection_per_rep : int, default=2000
        The (maximum) number of candidate features selected per representation.
        Only applied in two stages strategies (RS and SR).
    nsax : int, default=0
        The number of representations produced by SAX transformation.
    nsfa : int, default=5
        The number of representations produced by SFA transformation.

        Note: including any SFA transformations will prevent the estimator from being
        serialised (no pickling).
    custom_config : dict, default=None
        Customized parameters for the symbolic transformation.
    random_state : int, default=None
        If `int`, random_state is the seed used by the random number generator;
    sfa_norm : bool, default=True
        Time series normalisation (standardisation).

    Notes
    -----
    The `mrsqm` package uses a different license (GPL-3.0) from the aeon BSD3 license
    covering this interface wrapper.
    See https://github.com/mlgig/mrsqm for the original implementation.

    References
    ----------
    .. [1] Nguyen, Thach Le, and Georgiana Ifrim. "Fast time series classification with
        random symbolic subsequences." Advanced Analytics and Learning on Temporal Data:
        7th ECML PKDD Workshop, AALTD 2022, Grenoble, France, September 19–23, 2022.

    Examples
    --------
    >>> from aeon.classification.dictionary_based import MrSQMClassifier
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, y = make_example_3d_numpy(random_state=0)
    >>> clf = MrSQMClassifier(random_state=0) # doctest: +SKIP
    >>> clf.fit(X, y) # doctest: +SKIP
    MrSQMClassifier(...)
    >>> clf.predict(X) # doctest: +SKIP
    """

    _tags = {
        "X_inner_type": "numpy3D",
        "algorithm_type": "dictionary",
        "cant_pickle": True,
        "python_dependencies": "mrsqm",
    }

    def __init__(
        self,
        strat: str = "RS",
        features_per_rep: int = 500,
        selection_per_rep: int = 2000,
        nsax: int = 0,
        nsfa: int = 5,
        sfa_norm: bool = True,
        custom_config: Union[dict, None] = None,
        random_state: Union[int, None] = None,
    ) -> None:
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        self.nsax = nsax
        self.nsfa = nsfa
        self.sfa_norm = sfa_norm
        self.custom_config = custom_config
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        from mrsqm import MrSQMClassifier

        self.clf_ = MrSQMClassifier(
            strat=self.strat,
            features_per_rep=self.features_per_rep,
            selection_per_rep=self.selection_per_rep,
            nsax=self.nsax,
            nsfa=self.nsfa,
            sfa_norm=self.sfa_norm,
            custom_config=self.custom_config,
            random_state=self.random_state,
        )
        self.clf_.fit(X, y)

        return self

    def _predict(self, X) -> np.ndarray:
        return self.clf_.predict(X)

    def _predict_proba(self, X) -> np.ndarray:
        return self.clf_.predict_proba(X)

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
        return {
            "features_per_rep": 50,
            "selection_per_rep": 200,
            "nsax": 1,
            "nsfa": 1,
        }
