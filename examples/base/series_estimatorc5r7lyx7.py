# %%NBQA-CELL-SEPfc780c
import numpy as np
import pandas as pd

from aeon.base import BaseSeriesEstimator

# We use the abstract base class for example purposes, regular classes will not
# have a class axis parameter.
bs = BaseSeriesEstimator(axis=0)


# %%NBQA-CELL-SEPfc780c
# By default, "capability:multivariate" is False, axis is 0 and X_inner_type is
# np.ndarray
# With this config, the output should always be an np.ndarray shape (100, 1)
d1 = np.random.random(size=(100))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"1. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
# The axis parameter will not change the output shape of 1D inputs such as pd.Series
# or univariate np.ndarray
d1 = np.random.random(size=(100))
d2 = bs._preprocess_series(d1, axis=1, store_metadata=True)
print(
    f"2. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
# A 2D array with the channel axis of size 1 will produce the same result
d1 = np.random.random(size=(100, 1))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"3. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
# The shape used can be swapped, but the axis parameter must be set correctly
d1 = np.random.random(size=(1, 100))
d2 = bs._preprocess_series(d1, axis=1, store_metadata=True)
print(
    f"4. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
# Other types will be converted to X_inner_type
d1 = pd.Series(np.random.random(size=(100)))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"5. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
d1 = pd.DataFrame(np.random.random(size=(100, 1)))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"6. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)


# %%NBQA-CELL-SEPfc780c
# Passing a multivariate array will raise an error if capability:multivariate is False
d1 = np.random.random(size=(100, 5))
try:
    bs._preprocess_series(d1, axis=0, store_metadata=True)
except ValueError as e:
    print(f"7. {e}")


# %%NBQA-CELL-SEPfc780c
# The capability:multivariate tag must be set to True to work with multivariate series
# If the estimator does not have this tag, then the implementation cannot handle the
# input
bs = bs.set_tags(**{"capability:multivariate": True})
# Both of these can be True at the same time, but for examples sake we disable
# univariate
bs = bs.set_tags(**{"capability:univariate": False})


# %%NBQA-CELL-SEPfc780c
# axis 0 means each row is a time series
d1 = np.random.random(size=(100, 5))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"1. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)
n_channels = bs.metadata_["n_channels"]
print(f"n_channels: {n_channels}")


# %%NBQA-CELL-SEPfc780c
# axis 1 means each column is a time series. If the axis is set incorrectly, the
# output shape will be wrong
d1 = np.random.random(size=(100, 5))
d2 = bs._preprocess_series(d1, axis=1, store_metadata=True)
print(
    f"2. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)
n_channels = bs.metadata_["n_channels"]
print(f"n_channels: {n_channels}")


# %%NBQA-CELL-SEPfc780c
# Conversions work similar to univariate series, but there is more emphasis on correctly
# setting the axis parameter
d1 = pd.DataFrame(np.random.random(size=(100, 5)))
d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
print(
    f"3. Input type = {type(d1)}, input shape = {d1.shape}, "
    f"output type = {type(d2)}, output shape = {d2.shape}"
)
n_channels = bs.metadata_["n_channels"]
print(f"n_channels: {n_channels}")


# %%NBQA-CELL-SEPfc780c
# Passing a univariate array will raise an error if capability:univariate is False
d1 = pd.Series(np.random.random(size=(100,)))
try:
    d2 = bs._preprocess_series(d1, axis=0, store_metadata=True)
except ValueError as e:
    print(f"4. {e}")
