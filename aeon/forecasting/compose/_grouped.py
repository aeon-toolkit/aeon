"""Implements compositors for performing forecasting by group."""

from aeon.forecasting.base._delegate import _DelegatedForecaster
from aeon.utils._data_types import ALL_TIME_SERIES_TYPES
from aeon.utils.validation._input import _abstract_type

__maintainer__ = []
__all__ = ["ForecastByLevel"]


class ForecastByLevel(_DelegatedForecaster):
    """Forecast by instance or panel.

    Used to apply multiple copies of `forecaster` by instance or by panel.

    If `groupby="global"`, behaves like `forecaster`.
    If `groupby="local"`, fits a clone of `forecaster` per time series instance.
    If `groupby="panel"`, fits a clone of `forecaster` per panel (first non-time level).

    The fitted forecasters can be accessed in the `forecasters_` attribute,
    if more than one clone is fitted, otherwise in the `forecaster_` attribute.

    Parameters
    ----------
    forecaster : aeon forecaster used in ForecastByLevel
        A "blueprint" forecaster, state does not change when `fit` is called.
    groupby : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are grouped to fit clones of `forecaster`
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree

    Attributes
    ----------
    forecaster_ : aeon forecaster, present only if `groupby` is "global"
        clone of `forecaster` used for fitting and forecasting
    forecasters_ : pd.DataFrame of aeon forecaster, present otherwise
        entries are clones of `forecaster` used for fitting and forecasting

    Examples
    --------
    >>> from aeon.forecasting.naive import NaiveForecaster
    >>> from aeon.forecasting.compose import ForecastByLevel
    >>> from aeon.testing.utils.data_gen import _make_hierarchical
    >>> y = _make_hierarchical()
    >>> f = ForecastByLevel(NaiveForecaster(), groupby="local")
    >>> f.fit(y)
    ForecastByLevel(...)
    >>> fitted_forecasters = f.forecasters_
    >>> fitted_forecasters_alt = f.get_fitted_params()["forecasters"]
    """

    _tags = {
        "requires-fh-in-fit": False,
        "capability:missing_values": True,
        "y_input_type": "both",
        "y_inner_type": ALL_TIME_SERIES_TYPES,
        "X_inner_type": ALL_TIME_SERIES_TYPES,
        "fit_is_empty": False,
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    def __init__(self, forecaster, groupby="local"):
        self.forecaster = forecaster
        self.groupby = groupby

        self.forecaster_ = forecaster.clone()

        super().__init__()

        self.clone_tags(self.forecaster_)
        self.set_tags(**{"fit_is_empty": False})

        if groupby == "local":
            scitypes = ["Series"]
        elif groupby == "global":
            scitypes = ["Series", "Panel", "Hierarchical"]
        elif groupby == "panel":
            scitypes = ["Series", "Panel"]
        else:
            raise ValueError(
                "groupby in ForecastByLevel must be one of"
                ' "local", "global", "panel", '
                f"but found {groupby}"
            )

        internal_types = [
            x for x in ALL_TIME_SERIES_TYPES if _abstract_type(x) in scitypes
        ]

        # this ensures that we convert in the inner estimator
        # but vectorization/broadcasting happens at the level of groupby
        self.set_tags(**{"y_inner_type": internal_types})
        self.set_tags(**{"X_inner_type": internal_types})

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
        params : dict or list of dict
        """
        from aeon.forecasting.naive import NaiveForecaster

        groupbys = ["local", "panel", "global"]

        f = NaiveForecaster()

        params = [{"forecaster": f, "groupby": g} for g in groupbys]

        return params
