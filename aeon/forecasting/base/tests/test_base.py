"""Testing advanced functionality of the base class."""

__maintainer__ = []

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from aeon.forecasting.base._base import _format_moving_cutoff_predictions
from aeon.testing.utils.data_gen import make_example_3d_numpy, make_series
from aeon.utils.conversion import convert_collection
from aeon.utils.validation._dependencies import _check_soft_dependencies

COLLECTION_TYPES = ["pd-multiindex", "nested_univ", "numpy3D"]


def _get_y(input_type, n_cases):
    y, _ = make_example_3d_numpy(n_cases=n_cases, random_state=42)
    y = convert_collection(y, input_type)
    return y


def _get_n_cases_hierarchical(y):
    time_obj = y.reset_index(-1).drop(y.columns, axis=1)
    inst_inds = time_obj.index.unique()
    n_cases = len(inst_inds)
    return n_cases


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
    """Test the format_moving_cutoff_predictions function."""
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
