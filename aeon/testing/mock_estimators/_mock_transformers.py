from aeon.transformations.base import BaseTransformer


class MockTransformer(BaseTransformer):
    """Mock transformer to alter data.

    Parameters
    ----------
    power : int or float, default=0.5
        The power to raise the input timeseries to.

    Attributes
    ----------
    power : int or float
        User supplied power.
    """

    _tags = {
        "input_data_type": "Series",
        # what is the abstract type of X: Series, or Panel
        "output_data_type": "Series",
        "X_inner_type": ["pd.DataFrame", "pd.Series"],
        "fit_is_empty": True,
        "transform-returns-same-time-index": True,
        "capability:multivariate": True,
        "capability:inverse_transform": True,
    }

    def __init__(self, power=0.5):
        self.power = power

        if not isinstance(self.power, (int, float)):
            raise ValueError(
                f"Expected `power` to be int or float, but found {type(self.power)}."
            )
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version."""
        return X.pow(self.power)

    def _inverse_transform(self, X, y=None):
        """Reverse transformation on `X`."""
        return X.pow(1.0 / self.power)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"power": 2.5}
