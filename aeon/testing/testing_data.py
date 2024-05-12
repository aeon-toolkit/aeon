from aeon.base import BaseCollectionEstimator, BaseSeriesEstimator

TEST_DATA_DICT = {
    "UnivariateCollection": None,
    "MulrivariateCollection": None,
    "UnequalLengthCollection": None,
    "MissingValuesCollection": None,
    "UnivariateSeries": None,
    "MultivariateSeries": None,
    "MissingValuesSeries": None,
}


def get_data_types_for_estimator(estimator):
    """Get data types for estimator.

    Parameters
    ----------
    estimator : BaseEstimator instance or class
        Estimator instance or class to check for valid input data types.
    """
    univariate = estimator.get_tag("capability:univariate", True)
    multivariate = estimator.get_tag("capability:multivariate", False)
    unequal_length = estimator.get_tag("capability:unequal_length", False)
    missing_values = estimator.get_tag("capability:missing_values", False)
    datatypes = []

    if isinstance(estimator, BaseCollectionEstimator):
        if univariate:
            datatypes.append("UnivariateCollection")
        if multivariate:
            datatypes.append("MultivariateCollection")
        if unequal_length:
            datatypes.append("UnequalLengthCollection")
        if missing_values:
            datatypes.append("MissingValuesCollection")
    elif isinstance(estimator, BaseSeriesEstimator):
        if univariate:
            datatypes.append("UnivariateSeries")
        if multivariate:
            datatypes.append("MultivariateSeries")
        if missing_values:
            datatypes.append("MissingValuesSeries")
    else:
        raise ValueError(f"Unknown estimator type: {type(estimator)}")

    if len(datatypes) == 0:
        raise ValueError(f"No valid data types found for estimator {estimator}")

    return datatypes
