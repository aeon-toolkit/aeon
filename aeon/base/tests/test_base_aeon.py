"""Tests for BaseObject universal base class that require aeon or sklearn imports."""

__maintainer__ = []


def test_get_fitted_params_sklearn():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj aeon component returns expected nested params
    """
    from aeon.datasets import load_airline
    from aeon.forecasting.trend import TrendForecaster

    y = load_airline()
    f = TrendForecaster().fit(y)

    params = f.get_fitted_params()

    assert "regressor__coef" in params.keys()
    assert "regressor" in params.keys()
    assert "regressor__intercept" in params.keys()


def test_get_fitted_params_sklearn_nested():
    """Tests fitted parameter retrieval with sklearn components.

    Raises
    ------
    AssertionError if logic behind get_fitted_params is incorrect, logic tested:
        calling get_fitted_params on obj aeon component returns expected nested params
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from aeon.datasets import load_airline
    from aeon.forecasting.trend import TrendForecaster

    y = load_airline()
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    f = TrendForecaster(pipe)
    f.fit(y)

    params = f.get_fitted_params()

    assert "regressor" in params.keys()
    assert "regressor__n_features_in" in params.keys()
