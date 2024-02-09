"""Testing advanced functionality of the base class."""

__author__ = ["fkiraly"]

from functools import reduce
from operator import mul

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from aeon.datatypes import check_is_mtype, convert
from aeon.forecasting.arima import ARIMA
from aeon.forecasting.base._base import _format_moving_cutoff_predictions
from aeon.testing.utils.data_gen import (
    _make_hierarchical,
    make_example_3d_numpy,
    make_series,
)
from aeon.utils.conversion import convert_collection
from aeon.utils.index_functions import get_cutoff, get_window
from aeon.utils.validation._dependencies import _check_soft_dependencies

COLLECTION_TYPES = ["pd-multiindex", "nested_univ", "numpy3D"]
HIER_TYPES = ["pd_multiindex_hier"]


def _get_y(input_type, n_instances):
    y, _ = make_example_3d_numpy(n_cases=n_instances, random_state=42)
    y = convert_collection(y, input_type)
    return y


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("input_type", COLLECTION_TYPES)
def test_vectorization_series_to_panel(input_type):
    """Test that forecaster vectorization works for Panel data.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10
    y = _get_y(input_type, n_instances)
    f = ARIMA()
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, input_type, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of type {input_type}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg

    cutoff_expected = get_cutoff(y)
    msg = (
        "estimator in vectorization test does not properly update cutoff, "
        f"expected {cutoff_expected}, but found {f.cutoff}"
    )
    assert f.cutoff == cutoff_expected, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("input_type", HIER_TYPES)
def test_vectorization_series_to_hier(input_type):
    """Test that forecaster vectorization works for Hierarchical data.

    This test passes Hierarchical data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=input_type)

    f = ARIMA()
    y_pred = f.fit(y).predict([1, 2, 3])
    valid, _, metadata = check_is_mtype(y_pred, input_type, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {input_type}, using the ARIMA forecaster"
    )

    assert valid, msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg

    msg = (
        "estimator in vectorization test does not properly update cutoff, "
        f"expected {y}, but found {f.cutoff}"
    )
    assert f.cutoff == get_cutoff(y), msg


PROBA_DF_METHODS = ["predict_interval", "predict_quantiles", "predict_var"]


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
@pytest.mark.parametrize("input_type", COLLECTION_TYPES)
def test_vectorization_series_to_panel_proba(method, input_type):
    """Test that forecaster vectorization works for Panel data, predict_proba.

    This test passes Panel data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    n_instances = 10
    y = _get_y(input_type, n_instances)

    est = ARIMA().fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd-multiindex"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(y_pred, expected_mtype, return_metadata=True)

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {input_type}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
@pytest.mark.parametrize("input_type", HIER_TYPES)
def test_vectorization_series_to_hier_proba(method, input_type):
    """Test that forecaster vectorization works for Hierarchical data, predict_proba.

    This test passes Hierarchical data to the ARIMA forecaster which internally has an
    implementation for Series only, so the BaseForecaster has to vectorize.
    """
    hierarchy_levels = (2, 4)
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)
    y = convert(y, from_type="pd_multiindex_hier", to_type=input_type)

    est = ARIMA().fit(y)
    y_pred = getattr(est, method)([1, 2, 3])

    if method in ["predict_interval", "predict_quantiles"]:
        expected_mtype = method.replace("ict", "")
    elif method in ["predict_var"]:
        expected_mtype = "pd_multiindex_hier"
    else:
        RuntimeError(f"bug in test, unreachable state, method {method} queried")

    valid, _, _ = check_is_mtype(y_pred, expected_mtype, return_metadata=True)

    msg = (
        f"vectorization of forecaster method {method} does not work for test example "
        f"of mtype {input_type}, using the ARIMA forecaster"
    )

    assert valid, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("method", PROBA_DF_METHODS)
def test_vectorization_preserves_row_index_names(method):
    """Test that forecaster vectorization preserves row index names in forecast."""
    hierarchy_levels = (2, 4)
    y = _make_hierarchical(hierarchy_levels=hierarchy_levels, random_state=84)

    est = ARIMA().fit(y, fh=[1, 2, 3])
    y_pred = getattr(est, method)()

    msg = (
        f"vectorization of forecaster method {method} changes row index names, "
        f"but it shouldn't. Tested using the ARIMA forecaster."
    )

    assert y_pred.index.names == y.index.names, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none"),
    reason="skip test if required soft dependency for ARIMA not available",
)
@pytest.mark.parametrize("mtype", HIER_TYPES)
@pytest.mark.parametrize("exogeneous", [True, False])
def test_vectorization_multivariate(mtype, exogeneous):
    """Test that forecaster vectorization preserves row index names in forecast."""
    hierarchy_levels = (2, 4)
    n_instances = reduce(mul, hierarchy_levels)

    y = _make_hierarchical(
        hierarchy_levels=hierarchy_levels, random_state=84, n_columns=2
    )

    if exogeneous:
        y_fit = get_window(y, lag=pd.Timedelta("3D"))
        X_fit = y_fit
        X_pred = get_window(y, window_length=pd.Timedelta("3D"), lag=pd.Timedelta("0D"))
    else:
        y_fit = y
        X_fit = None
        X_pred = None

    est = ARIMA().fit(y=y_fit, X=X_fit, fh=[1, 2, 3])
    y_pred = est.predict(X=X_pred)
    valid, _, metadata = check_is_mtype(y_pred, mtype, return_metadata=True)

    msg = (
        f"vectorization of forecasters does not work for test example "
        f"of mtype {mtype}, using the ARIMA forecaster"
    )
    assert valid, msg

    msg = (
        "vectorization over variables produces wrong set of variables in predict, "
        f"expected {y_fit.columns}, found {y_pred.columns}"
    )
    assert set(y_fit.columns) == set(y_pred.columns), msg

    y_pred_instances = metadata["n_instances"]
    msg = (
        f"vectorization test produces wrong number of instances "
        f"expected {n_instances}, found {y_pred_instances}"
    )

    assert y_pred_instances == n_instances, msg

    y_pred_equal_length = metadata["is_equal_length"]
    msg = (
        "vectorization test produces non-equal length Panel forecast, should be "
        "equal length, and length equal to the forecasting horizon [1, 2, 3]"
    )
    assert y_pred_equal_length, msg


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_dynamic_tags_reset_properly():
    """Test that dynamic tags are being reset properly."""
    from aeon.forecasting.compose import MultiplexForecaster
    from aeon.forecasting.theta import ThetaForecaster
    from aeon.forecasting.var import VAR

    # this forecaster will have the y_input_type tag set to "univariate"
    f = MultiplexForecaster([("foo", ThetaForecaster()), ("var", VAR())])
    f.set_params(selected_forecaster="var")

    X_multivariate = make_series(n_columns=2)
    # fit should reset the estimator, and set y_input_type tag to "multivariate"
    # the fit will cause an error if this is not happening properly
    f.fit(X_multivariate)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_predict_residuals():
    """Test that predict_residuals has no side-effect."""
    from aeon.forecasting.base import ForecastingHorizon
    from aeon.forecasting.model_selection import temporal_train_test_split
    from aeon.forecasting.theta import ThetaForecaster

    y = make_series(n_columns=1)
    y_train, y_test = temporal_train_test_split(y)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12)
    forecaster.fit(y_train, fh=fh)

    y_pred_1 = forecaster.predict()
    y_resid = forecaster.predict_residuals()
    y_pred_2 = forecaster.predict()
    assert_series_equal(y_pred_1, y_pred_2)
    assert y_resid.index.equals(y_train.index)


def test_format_moving_cutoff_predictions():
    with pytest.raises(ValueError, match="`y_preds` must be a list"):
        _format_moving_cutoff_predictions("foo", "bar")
    with pytest.raises(
        ValueError, match="y_preds must be a list of pd.Series or pd.DataFrame"
    ):
        _format_moving_cutoff_predictions([1, 2, 3], "bar")
    res = pd.DataFrame([1, 2, 3])
    preds = [res, "foo"]
    with pytest.raises(
        ValueError, match="all elements of y_preds must be of the same type"
    ):
        _format_moving_cutoff_predictions(preds, "bar")
    res2 = pd.DataFrame([1, 2, 2, 3, 4, 4])
    preds = [res, res2]
    with pytest.raises(
        ValueError, match="all elements of y_preds must be of the same length"
    ):
        _format_moving_cutoff_predictions(preds, "bar")
    res = pd.DataFrame({"Column1": [1, 2, 3], "Column2": [3, 4, 5]})
    res2 = pd.DataFrame({"Col1": [1, 2, 3], "Col2": [3, 4, 5]})
    preds = [res, res2]
    with pytest.raises(
        ValueError, match="all elements of y_preds must have the same columns"
    ):
        _format_moving_cutoff_predictions(preds, "bar")
    res = pd.DataFrame({"Col1": [1, 2, 3], "Col2": [3, 4, 5]})
    preds = [res, res2]
    r = _format_moving_cutoff_predictions(preds, "bar")
    assert isinstance(r, pd.DataFrame)
