"""Utilities for loading datasets."""

__maintainer__ = []
__all__ = [
    "load_airline",
    "load_plaid",
    "load_arrow_head",
    "load_gunpoint",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_basic_motions",
    "load_japanese_vowels",
    "load_solar",
    "load_shampoo_sales",
    "load_longley",
    "load_lynx",
    "load_acsf1",
    "load_unit_test",
    "load_uschange",
    "load_PBS_dataset",
    "load_gun_point_segmentation",
    "load_electric_devices_segmentation",
    "load_unit_test_tsf",
    "load_covid_3month",
]

import os

import numpy as np
import pandas as pd

from aeon.datasets import load_from_tsf_file
from aeon.datasets._data_loaders import _load_saved_dataset, _load_tsc_dataset

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


def load_gunpoint(split=None, return_type="numpy3d"):
    """Load the GunPoint univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 1, 150) (return_type="numpy3d") or shape (n_cases,
        150) (return_type="numpy2d"), where n_cases is either 150 (split="train" or
        "test") or 300.
    y: np.ndarray
        1D array of length 150 or 300. The class labels for each time series instance
        in X.

    Examples
    --------
    >>> from aeon.datasets import load_gunpoint
    >>> X, y = load_gunpoint()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      150
    Train cases:        50
    Test cases:         150
    Number of classes:  2
    Details: https://timeseriesclassification.com/description.php?Dataset=GunPoint
    """
    return _load_tsc_dataset("GunPoint", split, return_type=return_type)


def load_osuleaf(split=None, return_type="numpy3d"):
    """Load the OSULeaf univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 1, 427) (return_type="numpy3d") or shape (n_cases,
        427) (return_type="numpy2d"), where n_cases where n_cases is either 200
        (split = "train") 242 (split="test") or 442.
    y: np.ndarray
        1D array of length 200, 242 or 542. The class labels for each time series
        instance in X.

    Examples
    --------
    >>> from aeon.datasets import load_osuleaf
    >>> X, y = load_osuleaf()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      427
    Train cases:        200
    Test cases:         242
    Number of classes:  6
    Details: https://www.timeseriesclassification.com/description.php?Dataset=OSULeaf
    """
    return _load_tsc_dataset("OSULeaf", split, return_type=return_type)


def load_italy_power_demand(split=None, return_type="numpy3d"):
    """Load ItalyPowerDemand univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 1, 24) (return_type="numpy3d") or shape (n_cases,
        24) (return_type="numpy2d"), where n_cases where n_cases is either 67
        (split = "train") 1029 (split="test") or 1096.
    y: np.ndarray
        1D array of length 67, 1029 or 1096. The class labels for each time series
        instance in X.

    Examples
    --------
    >>> from aeon.datasets import load_italy_power_demand
    >>> X, y = load_italy_power_demand()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2
    Details:https://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_tsc_dataset(name, split, return_type=return_type)


def load_unit_test(split=None, return_type="numpy3d"):
    """
    Load UnitTest data.

    This is an equal length univariate time series classification problem. It is a
    stripped down version of the ChinaTown problem that is used in correctness tests
    for classification.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure containing series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 1, 24) (return_type="numpy3d) or shape (n_cases,
        24) (return_type="numpy2d), where n_cases where n_cases is either 20
        (split = "train") 22 (split="test") or 42.
    y: np.ndarray
        1D array of length 20, 22 or 42
        The class labels for each time series instance in X

    Examples
    --------
    >>> from aeon.datasets import load_unit_test
    >>> X, y = load_unit_test()

    Details
    -------
    This is the Chinatown problem with a smaller test set, useful for rapid tests.
    Dimensionality:     univariate
    Series length:      24
    Train cases:        20
    Test cases:         22 (full dataset has 345)
    Number of classes:  2
    Details: https://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    """
    return _load_saved_dataset("UnitTest", split, return_type)


def load_arrow_head(split=None, return_type="numpy3d"):
    """
    Load the ArrowHead univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X:np.ndarray
        shape (n_cases, 1, 251) (if return_type="numpy3d") or shape (n_cases,
        251) (return_type="numpy2d"), where n_cases where n_cases is either 36
        (split = "train"), 175 (split="test") or 211.
    y: np.ndarray
        1D array of length 36, 175 or 211. The class labels for each time series
        instance in X.

    Examples
    --------
    >>> from aeon.datasets import load_arrow_head
    >>> X, y = load_arrow_head()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      251
    Train cases:        36
    Test cases:         175
    Number of classes:  3
    Details: https://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """
    return _load_saved_dataset(name="ArrowHead", split=split, return_type=return_type)


def load_acsf1(split=None, return_type="numpy3d"):
    """Load the ACSF1 univariate dataset on power consumption of typical appliances.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 1, 1460) (if return_type="numpy3d") or shape (n_cases,
        1460) (return_type="numpy2d"), where n_cases where n_cases is either 100
        (split = "train" or split="test") or 200.
    y: np.ndarray
        1D array of length 100 or 200. The class labels for each time series instance
        in X.

    Examples
    --------
    >>> from aeon.datasets import load_acsf1
    >>> X, y = load_acsf1()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      1460
    Train cases:        100
    Test cases:         100
    Number of classes:  10
    Details: https://www.timeseriesclassification.com/description.php?Dataset=ACSF1
    """
    return _load_tsc_dataset("ACSF1", split, return_type=return_type)


def load_basic_motions(split=None, return_type="numpy3d"):
    """
    Load the BasicMotions time series classification problem.

    Example of a multivariate problem with equal length time series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be "numpy3d" or "np-list".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: np.ndarray
        shape (n_cases, 6, 100) (if return_type="numpy3d"), where n_cases where
        n_cases is either 40 (split = "train" or split="test") or 80.
    y: np.ndarray
        1D array of length 40 or 80. The class labels for each time series instance
        in X.

    Notes
    -----
    Dimensionality:     multivariate, 6
    Series length:      100
    Train cases:        40
    Test cases:         40
    Number of classes:  4
    Details:https://www.timeseriesclassification.com/description.php?Dataset=BasicMotions
    """
    if return_type == "numpy2d" or return_type == "numpy2D":
        raise ValueError(
            f"BasicMotions loader: Error, attempting to load into a {return_type} "
            f"array, but cannot because it is a multivariate problem. Use "
            f"numpy3d instead"
        )
    return _load_saved_dataset(
        name="BasicMotions", split=split, return_type=return_type
    )


def load_plaid(split=None):
    """Load the PLAID univariate time series classification problem.

    Example of a univariate problem with unequal length time series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.

    Returns
    -------
    X: list of 2D np.ndarray, one for each series.
    y: 1D numpy array of length len(X). The class labels for each time series
    instance in X.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      variable
    Train cases:        537
    Test cases:         537
    Number of classes:  2
    Details: https://timeseriesclassification.com/description.php?Dataset=PLAID

    Examples
    --------
    >>> from aeon.datasets import load_plaid
    >>> X, y = load_plaid()
    """
    return _load_tsc_dataset("PLAID", split, return_type="np-list")


def load_japanese_vowels(split=None):
    """Load the JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal length series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.

    Returns
    -------
    X: list of 2D np.ndarray, one for each series.
    y: 1D numpy array of length len(X). The class labels for each time series instance
    in X.

    Examples
    --------
    >>> from aeon.datasets import load_japanese_vowels
    >>> X, y = load_japanese_vowels()

    Notes
    -----
    Dimensionality:     12
    Series length:      variable (7-29)
    Train cases:        270
    Test cases:         370
    Number of classes:  9
    Details: https://timeseriesclassification.com/description.php?Dataset=JapaneseVowels
    """
    return _load_tsc_dataset("JapaneseVowels", split, return_type="np-list")


def load_covid_3month(split=None, return_type="numpy3d"):
    """Load dataset of last three months confirmed covid cases.

    Parameters
    ----------
    split: None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By
        default, it loads both.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Returns
    -------
    X: np.ndarray
        The time series data.
    y: np.ndarray
        The regression values for each case in X.

    Examples
    --------
    >>> from aeon.datasets import load_covid_3month
    >>> X, y = load_covid_3month()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      84
    Train cases:        140
    Test cases:         61
    Number of classes:  -

    The goal of this dataset is to predict COVID-19's death rate on 1st April 2020 for
    each country using daily confirmed cases for the last three months. This dataset
    contains 201 time series with no missing values, where each time series is
    the daily confirmed cases for a country.
    The data was obtained from WHO's COVID-19 database.
    Please refer to https://covid19.who.int/ for more details

    Dataset details: https://zenodo.org/record/3902690#.Yy1z_HZBxEY
    =Covid3Month
    """
    name = "Covid3Month"
    X, y = _load_tsc_dataset(name, split, return_type)
    y = y.astype(float)
    return X, y


def load_cardano_sentiment(split=None, return_type="numpy3d"):
    """Load dataset of historical sentiment data for Cardano cryptocurrency.

    Parameters
    ----------
    split: None or str{"train", "test"}, default=None
        Whether to load the train or test partition of the problem. By
        default, it loads both.
    return_type: string, default="numpy3d"
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Returns
    -------
    X: np.ndarray
        The time series data.
    y: np.ndarray
        The regression values for each case in X.

    Examples
    --------
    >>> from aeon.datasets import load_cardano_sentiment
    >>> X, y = load_cardano_sentiment()

    Notes
    -----
    Dimensionality:     multivariate
    Series length:      24
    Train cases:        74
    Test cases:         33
    Number of classes:  -

    By combining historical sentiment data for Cardano cryptocurrency, extracted from
    EODHistoricalData and made available on Kaggle, with historical price data for the
    same cryptocurrency, extracted from CryptoDataDownload, we created the
    CardanoSentiment dataset, with 107 instances. The predictors are hourly close price
    (in USD) and traded volume during a day, resulting in 2-dimensional time series of
    length 24. The response variable is the normalized sentiment score on the day
    spanned by the timepoints.

    EODHistoricalData: https://perma.cc/37GN-BMRL
    CryptoDataDownload: https://perma.cc/4M79-7QY4
    Dataset details: https://arxiv.org/pdf/2305.01429.pdf
    """
    X, y = _load_tsc_dataset("CardanoSentiment", split, return_type)
    y = y.astype(float)
    return X, y


def load_gun_point_segmentation():
    """Load the GunPoint time series segmentation problem and returns X.

    We group TS of the UCR GunPoint dataset by class label and concatenate
    all TS to create segments with repeating temporal patterns and
    characteristics. The location at which different classes were
    concatenated are marked as change points.

    We resample the resulting TS to control the TS resolution.
    The window sizes for these datasets are hand-selected to capture
    temporal patterns but are approximate and limited to the values
    [10,20,50,100] to avoid over-fitting.

    Returns
    -------
    X : pd.Series
        Single time series for segmentation
    period_length : int
        The annotated period length by a human expert
    change_points : numpy array
        The change points annotated within the dataset

    Examples
    --------
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, period_length, change_points = load_gun_point_segmentation()
    """
    dir = "segmentation"
    name = "GunPoint"
    fname = name + ".csv"

    period_length = int(10)
    change_points = np.int32([900])

    path = os.path.join(MODULE, DIRNAME, dir, fname)
    ts = pd.read_csv(path, index_col=0, header=None).squeeze("columns")

    return ts, period_length, change_points


def load_electric_devices_segmentation():
    """Load the Electric Devices segmentation problem and returns X.

    We group TS of the UCR Electric Devices dataset by class label and concatenate
    all TS to create segments with repeating temporal patterns and
    characteristics. The location at which different classes were
    concatenated are marked as change points.

    We resample the resulting TS to control the TS resolution.
    The window sizes for these datasets are hand-selected to capture
    temporal patterns but are approximate and limited to the values
    [10,20,50,100] to avoid over-fitting.

    Returns
    -------
    X : pd.Series
        Single time series for segmentation
    period_length : int
        The annotated period length by a human expert
    change_points : numpy array
        The change points annotated within the dataset

    Examples
    --------
    >>> from aeon.datasets import load_electric_devices_segmentation
    >>> X, period_length, change_points = load_electric_devices_segmentation()
    """
    dir = "segmentation"
    name = "ElectricDevices"
    fname = name + ".csv"

    period_length = int(10)
    change_points = np.int32([1090, 4436, 5712, 7923])

    path = os.path.join(MODULE, DIRNAME, dir, fname)
    ts = pd.read_csv(path, index_col=0, header=None).squeeze("columns")

    return ts, period_length, change_points


def load_unit_test_tsf(return_type="tsf_default"):
    """
    Load tsf UnitTest dataset.

    Parameters
    ----------
    return_type : str - "pd_multiindex_hier" or "tsf_default" (default)
        - "tsf_default" = container that faithfully mirrors tsf format from the original
            implementation in: https://github.com/rakshitha123/TSForecasting/
            blob/master/utils/data_loader.py.

    Returns
    -------
    loaded_data: pd.DataFrame
        The converted dataframe containing the time series.
    frequency: str
        The frequency of the dataset.
    forecast_horizon: int
        The expected forecast horizon of the dataset.
    contain_missing_values: bool
        Whether the dataset contains missing values or not.
    contain_equal_length: bool
        Whether the series have equal lengths or not.
    """
    path = os.path.join(MODULE, DIRNAME, "UnitTest", "UnitTest_Tsf_Loader.tsf")
    data, meta = load_from_tsf_file(path, return_type=return_type)
    return (
        data,
        meta["frequency"],
        meta["forecast_horizon"],
        meta["contain_missing_values"],
        meta["contain_equal_length"],
    )


# forecasting data sets
def load_shampoo_sales(return_array=True):
    """Load the shampoo sales univariate time series dataset for forecasting.

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.Series.

    Returns
    -------
    np.ndarray or pd.Series
        Shampoo sales dataset

    Examples
    --------
    >>> from aeon.datasets import load_shampoo_sales
    >>> y = load_shampoo_sales()
    >>> type(y)
    <class 'numpy.ndarray'>
    >>> y = load_shampoo_sales(return_array=False)
    >>> type(y)
    <class 'pandas.core.series.Series'>

    Notes
    -----
    This dataset describes the monthly number of sales of shampoo over a 3
    year period. The units are a sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1

    References
    ----------
    .. [1] Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods
    and applications,
        John Wiley & Sons: New York. Chapter 3.
    """
    name = "ShampooSales"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
    if return_array:
        return y.values
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of shampoo sales"
    return y


def load_lynx(return_array=True):
    """Load the lynx univariate time series dataset for forecasting.

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.Series.

    Returns
    -------
    np.ndarray or pd.Series/DataFrame
        Lynx sales dataset

    Examples
    --------
    >>> from aeon.datasets import load_lynx
    >>> y = load_lynx()
    >>> type(y)
    <class 'numpy.ndarray'>
    >>> y = load_lynx(return_array=False)
    >>> type(y)
    <class 'pandas.core.series.Series'>

    Notes
    -----
    The annual numbers of lynx trappings for 1821–1934 in Canada. This
    time-series records the number of skins of
    predators (lynx) that were collected over several years by the Hudson's
    Bay Company. The dataset was
    taken from Brockwell & Davis (1991) and appears to be the series
    considered by Campbell & Walker (1977).

    Dimensionality:     univariate
    Series length:      114
    Frequency:          Yearly
    Number of cases:    1

    This data shows aperiodic, cyclical patterns, as opposed to periodic,
    seasonal patterns.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988). The New S
    Language. Wadsworth & Brooks/Cole.

    .. [2] Campbell, M. J. and Walker, A. M. (1977). A Survey of statistical
    work on the Mackenzie River series of
    annual Canadian lynx trappings for the years 1821–1934 and a new
    analysis. Journal of the Royal Statistical Society
    series A, 140, 411–431.
    """
    name = "Lynx"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
    if return_array:
        return y.values
    y.index = pd.PeriodIndex(y.index, freq="Y", name="Period")
    y.name = "Number of Lynx trappings"
    return y


def load_airline(return_array=True):
    """Load the airline univariate time series dataset [1].

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.Series.

    Returns
    -------
    np.ndarray or pd.Series
        Airline time series

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()
    >>> type(y)
    <class 'numpy.ndarray'>
    >>> y = load_airline(return_array=False)
    >>> type(y)
    <class 'pandas.core.series.Series'>

    Notes
    -----
    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Dimensionality:     univariate
    Series length:      144
    Frequency:          Monthly
    Number of cases:    1

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    .. [1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series
          Analysis, Forecasting and Control. Third Edition. Holden-Day.
          Series G.
    """
    name = "Airline"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
    if return_array:
        return y.values
    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of airline passengers"
    return y


def load_solar(return_array=True):
    """Get national solar estimates for GB from Sheffield Solar PV_Live API.

    This function calls the Sheffield Solar PV_Live API to extract national solar data
    for the GB eletricity network. Note that these are estimates of the true solar
    generation, since the true values are "behind the meter" and essentially
    unknown.

    The returned time series is half hourly. For more information please refer
    to [1]_.

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.Series.

    Returns
    -------
    np.ndarray or pd.Series
        Example Sheffield solar time series

    References
    ----------
    .. [1] https://www.solar.sheffield.ac.uk/pvlive/

    Examples
    --------
    >>> from aeon.datasets import load_solar  # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    """
    name = "solar"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, parse_dates=["datetime_gmt"], dtype={1: float})
    y = y.asfreq("30min")
    y = y.squeeze("columns")
    if return_array:
        return y.values
    return y


def load_PBS_dataset(return_array=True):
    """Load the Pharmaceutical Benefit Scheme univariate time series dataset [1]_.

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.Series.

    Returns
    -------
    np.ndarray or pd.Series
        PBS time series

    Examples
    --------
    >>> from aeon.datasets import load_PBS_dataset
    >>> y = load_PBS_dataset()

    Notes
    -----
    The Pharmaceutical Benefits Scheme (PBS) is the Australian government drugs
    subsidy scheme.
    Data comprises of the numbers of scripts sold each month for immune sera
    and immunoglobulin products in Australia.


    Dimensionality:     univariate
    Series length:      204
    Frequency:          Monthly
    Number of cases:    1

    The time series is intermittent, i.e contains small counts,
    with many months registering no sales at all,
    and only small numbers of items sold in other months.

    References
    ----------
    .. [1] Data for "Forecasting: Principles and Practice" (3rd Edition)
    """
    name = "PBS_dataset"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, dtype={1: float}).squeeze("columns")
    if return_array:
        return y.values
    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of scripts"
    return y


def load_uschange(return_array=True):
    """Load US Change forecasting dataset.

    An example of a single multivariate time series. The data is the percentage
    changes in quarterly personal consumption expenditure, personal disposable
    income, production, savings and the unemployment rate for the US, 1960 to 2016.

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    Channels:    ['Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      187
    Frequency:          Quarterly

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.DataFrame in wide format.

    Returns
    -------
    np.ndarray or pd.DataFrame
        US Change dataset, shape (5,187).

    Examples
    --------
    >>> from aeon.datasets import load_uschange
    >>> data = load_uschange()
    >>> data.shape
    (5, 187)
    >>> data = load_uschange(return_array=False)
    >>> data.shape
    (5, 187)

    References
    ----------
    .. [1] Data for "Forecasting: Principles and Practice" (2nd Edition)
    """
    name = "Uschange"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0).squeeze("columns")
    data = data.sort_values("Quarter")
    data = data.reset_index(drop=True)
    data.index = pd.Index(data.index, dtype=int)
    data.name = name
    data = data.drop("Quarter", axis=1)
    if return_array:
        return data.to_numpy().T
    return data.T


def load_longley(return_array=True):
    """Load the Longley multivariate time series.

    This time series contains six US macroeconomic
    variables from 1947 to 1962 that are known to be highly collinear.

    Dimensionality:     multivariate, 6
    Series length:      16
    Frequency:          Yearly
    Number of cases:    1

    Variable description:

    TOTEMP - Total employment
    GNPDEFL - Gross national product deflator
    GNP - Gross national product
    UNEMP - Number of unemployed
    ARMED - Size of armed forces
    POP - Population

    Parameters
    ----------
    return_array : bool, default=True
        return series as an np.ndarray if True, else as a pd.DataFrame in wide format.

    Returns
    -------
    np.ndarray or pd.DataFrame
        US Change dataset, shape (6, 16).

    Examples
    --------
    >>> from aeon.datasets import load_longley
    >>> data = load_longley()
    >>> data.shape
    (6, 16)
    >>> data = load_longley(return_array=False)
    >>> data.shape
    (6, 16)

    References
    ----------
    .. [1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Computer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """
    name = "Longley"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0)
    data = data.set_index("YEAR")
    data.index = pd.PeriodIndex(data.index, freq="Y", name="Period")
    data = data.astype(float)
    if return_array:
        return data.to_numpy().T
    return data.T
