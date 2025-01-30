"""Symbolic Fourier Approximation (SFA) Transformer.

Configurable SFA transform for discretising time series into words.

"""

__maintainer__ = []
__all__ = ["SFAWhole"]

from aeon.transformations.collection.dictionary_based import SFAFast


class SFAWhole(SFAFast):
    """Symbolic Fourier Approximation (SFA) Transformer.

    A whole series transform for SFA which holds the lower bounding lemma, see [1].

    It is implemented as a wrapper for the SFA-Fast transformer, the latter implements
    subsequence-based SFA extraction.

    This wrapper reduces non-needed parameters, and sets some usefull defaults for
    lower bounding.

    Parameters
    ----------
    word_length : int, default = 8
        Length of word to shorten window to (using DFT).
    alphabet_size : int, default = 4
        Number of values to discretise each value to.
    norm : boolean, default = False
        Mean normalise words by dropping first fourier coefficient.
    binning_method : str, default="equi-depth"
        The binning method used to derive the breakpoints. One of {"equi-depth",
        "equi-width", "information-gain", "information-gain-mae", "kmeans", "quantile"},
    variance : boolean, default = False
        If True, the Fourier coefficient selection is done via the largest variance.
        If False, the first Fourier coefficients are selected. Only applicable if
        labels are given.
    sampling_factor : float, default = None
       If set to a value <1.0, this percentage of samples are used to learn MCB bins.
    n_jobs : int, default = 1
        The number of jobs to run in parallel for both `transform`.
        ``-1`` means using all processors.

    Attributes
    ----------
    breakpoints: = []
    num_insts = 0
    num_atts = 0


    References
    ----------
    .. [1] Schäfer, Patrick, and Mikael Högqvist. "SFA: a symbolic fourier approximation
    and  index for similarity search in high dimensional datasets." Proceedings of the
    15th international conference on extending database technology. 2012.
    """

    _tags = {
        "requires_y": False,  # SFA is unsupervised for equi-depth and equi-width bins
        "capability:multithreading": True,
        "algorithm_type": "dictionary",
    }

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        norm=True,
        binning_method="equi-depth",
        variance=True,
        sampling_factor=None,
        random_state=None,
        n_jobs=1,
    ):
        super().__init__(
            word_length=word_length,
            alphabet_size=alphabet_size,
            norm=norm,
            binning_method=binning_method,
            variance=variance,
            sampling_factor=sampling_factor,
            random_state=random_state,
            n_jobs=n_jobs,
            # Default values for other parameters
            lower_bounding_distances=True,
            feature_selection="none",
            anova=False,
            save_words=False,
            lower_bounding=False,
            bigrams=False,
            skip_grams=False,
            remove_repeat_words=False,
            return_sparse=False,
            window_size=None,  # set in fit
        )

    def _fit_transform(self, X, y=None):
        super()._fit_transform(X, y, return_bag_of_words=False)
        return self.transform_words(X)

    def _fit(self, X, y=None):
        """Calculate word breakpoints.

        Parameters
        ----------
        X : 3d numpy array, input time series.
        y : array_like, target values (optional, ignored).

        Returns
        -------
        self: object
        """
        super()._fit_transform(X, y, return_bag_of_words=False)
        return self

    def _transform(self, X, y=None):
        """Transform data into SFA words.

        Parameters
        ----------
        X : 3d numpy array, input time series.

        Returns
        -------
        List of words containing SFA words
        """
        return self.transform_words(X)

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        # small window size for testing
        params = {
            "word_length": 4,
            "alphabet_size": 4,
            "variance": False,
        }
        return params
