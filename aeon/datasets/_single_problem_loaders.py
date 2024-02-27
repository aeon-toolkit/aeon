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
    "load_macroeconomic",
    "load_unit_test_tsf",
    "load_covid_3month",
]

import os
from urllib.error import HTTPError, URLError
from warnings import warn

import numpy as np
import pandas as pd

from aeon.datasets import load_from_tsf_file
from aeon.datasets._data_loaders import _load_saved_dataset, _load_tsc_dataset
from aeon.utils.validation._dependencies import _check_soft_dependencies

DIRNAME = "data"
MODULE = os.path.dirname(__file__)


def load_gunpoint(split=None, return_X_y=True, return_type="numpy3d"):
    """Load the GunPoint univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 150 or 300, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://timeseriesclassification.com/description.php?Dataset=GunPoint
    """
    return _load_tsc_dataset("GunPoint", split, return_X_y, return_type=return_type)


def load_osuleaf(split=None, return_X_y=True, return_type="numpy3d"):
    """Load the OSULeaf univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 200, 242 or 542, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://www.timeseriesclassification.com/description.php?Dataset=OSULeaf
    """
    return _load_tsc_dataset("OSULeaf", split, return_X_y, return_type=return_type)


def load_italy_power_demand(split=None, return_X_y=True, return_type="numpy3d"):
    """Load ItalyPowerDemand univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 67, 1029 or 1096, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.


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
    Details:http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """
    name = "ItalyPowerDemand"
    return _load_tsc_dataset(name, split, return_X_y, return_type=return_type)


def load_unit_test(split=None, return_X_y=True, return_type="numpy3d"):
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
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 20, 22 or 42, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://timeseriesclassification.com/description.php?Dataset=Chinatown
    for the full dataset
    """
    return _load_saved_dataset("UnitTest", split, return_X_y, return_type)


def load_arrow_head(split=None, return_X_y=True, return_type="numpy3d"):
    """
    Load the ArrowHead univariate time series classification problem.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 36, 175 or 211, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """
    return _load_saved_dataset(
        name="ArrowHead", split=split, return_X_y=return_X_y, return_type=return_type
    )


def load_acsf1(split=None, return_X_y=True, return_type="numpy3d"):
    """Load the ACSF1 univariate dataset on power consumption of typical appliances.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 100 or 200 only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://www.timeseriesclassification.com/description.php?Dataset=ACSF1
    """
    return _load_tsc_dataset("ACSF1", split, return_X_y, return_type=return_type)


def load_basic_motions(split=None, return_X_y=True, return_type="numpy3d"):
    """
    Load the BasicMotions time series classification problem.

    Example of a multivariate problem with equal length time series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, optional (default="numpy3d")
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
        1D array of length 40 or 80, only returned if return_X_y is True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

    Notes
    -----
    Dimensionality:     multivariate, 6
    Series length:      100
    Train cases:        40
    Test cases:         40
    Number of classes:  4
    Details:http://www.timeseriesclassification.com/description.php?Dataset=BasicMotions
    """
    if return_type == "numpy2d" or return_type == "numpy2D":
        raise ValueError(
            f"BasicMotions loader: Error, attempting to load into a {return_type} "
            f"array, but cannot because it is a multivariate problem. Use "
            f"numpy3d instead"
        )
    return _load_saved_dataset(
        name="BasicMotions", split=split, return_X_y=return_X_y, return_type=return_type
    )


def load_plaid(split=None, return_X_y=True, return_type="np-list"):
    """Load the PLAID univariate time series classification problem.

    Example of a univariate problem with unequal length time series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", default=None
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, default=True
        If True, returns (features, target) separately instead of as single data
        structure.
    return_type: string, default="np-list"
        Data structure to use for time series, should be "nested_univ" or "np-list".

    Raises
    ------
    ValueError is raised if the data cannot be stored in the requested return_type.

    Returns
    -------
    X: list of 2D np.ndarray, one for each series.
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

    Notes
    -----
    Dimensionality:     univariate
    Series length:      variable
    Train cases:        537
    Test cases:         537
    Number of classes:  2
    Details: http://timeseriesclassification.com/description.php?Dataset=PLAID

    Examples
    --------
    >>> from aeon.datasets import load_plaid
    >>> X, y = load_plaid()
    """
    return _load_tsc_dataset("PLAID", split, return_X_y, return_type=return_type)


def load_japanese_vowels(split=None, return_X_y=True, return_type="np-list"):
    """Load the JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal length series.

    Parameters
    ----------
    split: None or one of "TRAIN", "TEST", optional (default=None)
        Whether to load the train or test instances of the problem. By default it
        loads both train and test instances into a single array.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for features and the target.
    return_type: string, default="np-list"
        Data structure to use for time series, should be "nested_univ" or "np-list".

    Returns
    -------
    X: np.Pandas dataframe with 12 columns and a pd.Series in each cell
    y: 1D numpy array of length n, only returned if return_X_y if True
        The class labels for each time series instance in X
        If return_X_y is False, y is appended to X instead.

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
    Details: http://timeseriesclassification.com/description.php?Dataset=JapaneseVowels
    """
    return _load_tsc_dataset(
        "JapaneseVowels", split, return_X_y, return_type=return_type
    )


# forecasting data sets
def load_shampoo_sales():
    """Load the shampoo sales univariate time series dataset for forecasting.

    Returns
    -------
    y : pd.Series/DataFrame
        Shampoo sales dataset

    Examples
    --------
    >>> from aeon.datasets import load_shampoo_sales
    >>> y = load_shampoo_sales()

    Notes
    -----
    This dataset describes the monthly number of sales of shampoo over a 3
    year period.
    The units are a sales count.

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
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of shampoo sales"
    return y


def load_longley(y_name="TOTEMP"):
    """Load the Longley dataset for forecasting with exogenous variables.

    Parameters
    ----------
    y_name: str, optional (default="TOTEMP")
        Name of target variable (y)

    Returns
    -------
    y: pd.Series
        The target series to be predicted.
    X: pd.DataFrame
        The exogenous time series data for the problem.

    Examples
    --------
    >>> from aeon.datasets import load_longley
    >>> y, X = load_longley()

    Notes
    -----
    This mulitvariate time series dataset contains various US macroeconomic
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

    # Get target series
    y = data.pop(y_name)
    return y, data


def load_lynx():
    """Load the lynx univariate time series dataset for forecasting.

    Returns
    -------
    y : pd.Series/DataFrame
        Lynx sales dataset

    Examples
    --------
    >>> from aeon.datasets import load_lynx
    >>> y = load_lynx()

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
    y.index = pd.PeriodIndex(y.index, freq="Y", name="Period")
    y.name = "Number of Lynx trappings"
    return y


def load_airline():
    """Load the airline univariate time series dataset [1].

    Returns
    -------
    y : pd.Series
        Time series

    Examples
    --------
    >>> from aeon.datasets import load_airline
    >>> y = load_airline()

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

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of airline passengers"
    return y


def load_uschange(y_name="Consumption"):
    """Load MTS dataset for forecasting Growth rates of personal consumption and income.

    Returns
    -------
    y : pd.Series
        selected column, default consumption
    X : pd.DataFrame
        columns with explanatory variables

    Examples
    --------
    >>> from aeon.datasets import load_uschange
    >>> y, X = load_uschange()

    Notes
    -----
    Percentage changes in quarterly personal consumption expenditure,
    personal disposable income, production, savings and the
    unemployment rate for the US, 1960 to 2016.


    Dimensionality:     multivariate
    Columns:            ['Quarter', 'Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      188
    Frequency:          Quarterly
    Number of cases:    1

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    .. [1] Data for "Forecasting: Principles and Practice" (2nd Edition)
    """
    name = "Uschange"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    data = pd.read_csv(path, index_col=0).squeeze("columns")

    # Sort by Quarter then set simple numeric index
    # TODO add support for period/datetime indexing
    # data.index = pd.PeriodIndex(data.index, freq='Y')
    data = data.sort_values("Quarter")
    data = data.reset_index(drop=True)
    data.index = pd.Index(data.index, dtype=int)
    data.name = name
    y = data[y_name]
    if y_name != "Quarter":
        data = data.drop("Quarter", axis=1)
    X = data.drop(y_name, axis=1)
    return y, X


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


def load_PBS_dataset():
    """Load the Pharmaceutical Benefit Scheme univariate time series dataset [1]_.

    Returns
    -------
    y : pd.Series
     Time series

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

    # make sure time index is properly formatted
    y.index = pd.PeriodIndex(y.index, freq="M", name="Period")
    y.name = "Number of scripts"
    return y


def load_macroeconomic():
    """
    Load the US Macroeconomic Data [1]_.

    Returns
    -------
    y : pd.DataFrame
     Time series

    Examples
    --------
    >>> from aeon.datasets import load_macroeconomic
    >>> y = load_macroeconomic()  # doctest: +SKIP

    Notes
    -----
    US Macroeconomic Data for 1959Q1 - 2009Q3.

    Dimensionality:     multivariate, 14
    Series length:      203
    Frequency:          Quarterly
    Number of cases:    1

    This data is kindly wrapped via `statsmodels.datasets.macrodata`.

    References
    ----------
    .. [1] Wrapped via statsmodels:
          https://www.statsmodels.org/dev/datasets/generated/macrodata.html
    .. [2] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve
          Bank of St. Louis; http://research.stlouisfed.org/fred2/;
          accessed December 15, 2009.
    .. [3] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;
          http://www.bls.gov/data/; accessed December 15, 2009.
    """
    _check_soft_dependencies("statsmodels")
    import statsmodels.api as sm

    y = sm.datasets.macrodata.load_pandas().data
    y["year"] = y["year"].astype(int).astype(str)
    y["quarter"] = y["quarter"].astype(int).astype(str).apply(lambda x: "Q" + x)
    y["time"] = y["year"] + "-" + y["quarter"]
    y.index = pd.PeriodIndex(data=y["time"], freq="Q", name="Period")
    y = y.drop(columns=["year", "quarter", "time"])
    y.name = "US Macroeconomic Data"
    return y


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


def load_solar(
    start="2021-05-01",
    end="2021-09-01",
    normalise=True,
    return_full_df=False,
    api_version="v4",
):
    """Get national solar estimates for GB from Sheffield Solar PV_Live API.

    This function calls the Sheffield Solar PV_Live API to extract national solar data
    for the GB eletricity network. Note that these are estimates of the true solar
    generation, since the true values are "behind the meter" and essentially
    unknown.

    The returned time series is half hourly. For more information please refer
    to [1, 2]_.

    Parameters
    ----------
    start : string, default="2021-05-01"
        The start date of the time-series in "YYYY-MM-DD" format
    end : string, default="2021-09-01"
        The end date of the time-series in "YYYY-MM-DD" format
    normalise : boolean, default=True
        Normalise the returned time-series by installed capacity?
    return_full_df : boolean, default=False
        Return a pd.DataFrame with power, capacity, and normalised estimates?
    api_version : string or None, default="v4"
        API version to call. If None then a stored sample of the data is loaded.

    Return
    ------
    pd.Series

    References
    ----------
    .. [1] https://www.solar.sheffield.ac.uk/pvlive/
    .. [2] https://www.solar.sheffield.ac.uk/pvlive/api/

    Examples
    --------
    >>> from aeon.datasets import load_solar  # doctest: +SKIP
    >>> y = load_solar()  # doctest: +SKIP
    """
    name = "solar"
    fname = name + ".csv"
    path = os.path.join(MODULE, DIRNAME, name, fname)
    y = pd.read_csv(path, index_col=0, parse_dates=["datetime_gmt"], dtype={1: float})
    y = y.asfreq("30T")
    y = y.squeeze("columns")
    if api_version is None:
        return y

    def _load_solar(
        start="2021-05-01",
        end="2021-09-01",
        normalise=True,
        return_full_df=False,
        api_version="v4",
    ):
        """Private loader, for decoration with backoff."""
        url = (
            f"https://api0.solar.sheffield.ac.uk/pvlive/api/"
            f"{api_version}/gsp/0?start={start}T00:00:00&end={end}"
            f"extra_fields=capacity_mwp&data_format=csv"
        )
        df = (
            pd.read_csv(
                url, index_col=["gsp_id", "datetime_gmt"], parse_dates=["datetime_gmt"]
            )
            .droplevel(0)
            .sort_index()
        )
        df = df.asfreq("30T")
        df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]

        if return_full_df:
            df["generation_pu"] = df["generation_mw"] / df["capacity_mwp"]
            return df
        else:
            if normalise:
                return df["generation_pu"].rename("solar_gen")
            else:
                return df["generation_mw"].rename("solar_gen")

    try:
        return _load_solar(
            start=start,
            end=end,
            normalise=normalise,
            return_full_df=return_full_df,
            api_version=api_version,
        )
    except (URLError, HTTPError):
        warn(
            """
                    Error detected using API. Check connection, input arguments, and
                    API status here https://www.solar.sheffield.ac.uk/pvlive/api/.
                    Loading stored sample data instead.
                    """
        )
        return y


def load_covid_3month(split=None, return_X_y=True, return_type="numpy3d"):
    """Load dataset of last three months confirmed covid cases.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default, it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.
    return_type: string, optional (default="numpy3d")
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The regression values for each case in X

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
    if return_X_y:
        X, y = _load_tsc_dataset(name, split, return_X_y, return_type)
        y = y.astype(float)
        return X, y
    else:
        X = _load_tsc_dataset(name, split, return_X_y, return_type)
        return X


def load_cardano_sentiment(split=None, return_X_y=True, return_type="numpy3d"):
    """Load dataset of historical sentiment data for Cardano cryptocurrency.

    Parameters
    ----------
    split: None or str{"train", "test"}, optional (default=None)
        Whether to load the train or test partition of the problem. By
        default, it loads both.
    return_X_y: bool, optional (default=True)
        If True, returns (features, target) separately instead of a single
        dataframe with columns for
        features and the target.
    return_type: string, optional (default="numpy3d")
        Data structure to use for time series, should be either "numpy2d" or "numpy3d".

    Returns
    -------
    X: pd.DataFrame with m rows and c columns
        The time series data for the problem with m cases and c dimensions
    y: numpy array
        The regression values for each case in X

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
    name = "CardanoSentiment"
    if return_X_y:
        X, y = _load_tsc_dataset(name, split, return_X_y, return_type)
        y = y.astype(float)
        return X, y
    else:
        X = _load_tsc_dataset(name, split, return_X_y, return_type)
        return X
