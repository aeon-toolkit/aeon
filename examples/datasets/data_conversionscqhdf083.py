# %%NBQA-CELL-SEPfc780c
import numpy as np

from aeon.datatypes import convert

numpyarray = np.random.random(size=(100, 1))
series = convert(numpyarray, from_type="np.ndarray", to_type="xr.DataArray")
type(series)


# %%NBQA-CELL-SEPfc780c
from aeon.datatypes._series._convert import (
    convert_mvs_to_dask_as_series,
    convert_Mvs_to_xrdatarray_as_Series,
    convert_np_to_MvS_as_Series,
)

pd_dataframe = convert_np_to_MvS_as_Series(numpyarray)
type(pd_dataframe)


# %%NBQA-CELL-SEPfc780c
dask_dataframe = convert_mvs_to_dask_as_series(pd_dataframe)
type(dask_dataframe)


# %%NBQA-CELL-SEPfc780c
xrarray = convert_Mvs_to_xrdatarray_as_Series(pd_dataframe)
type(xrarray)


# %%NBQA-CELL-SEPfc780c
from aeon.utils.conversion import convert_collection

# 10 multivariate time series with 3 channels of length 100 in "numpy3D" format
multi = np.random.random(size=(10, 3, 100))
np_list = convert_collection(multi, output_type="np-list")
print(
    f" Type = {type(np_list)}, type first {type(np_list[0])} shape first "
    f"{np_list[0].shape}"
)


# %%NBQA-CELL-SEPfc780c
df_list = convert_collection(multi, output_type="df-list")
print(
    f" Type = {type(df_list)}, type first {type(df_list[0])} shape first "
    f"{df_list[0].shape}"
)


# %%NBQA-CELL-SEPfc780c
from aeon.utils.conversion._convert_collection import _from_numpy3d_to_pd_multiindex

mi = _from_numpy3d_to_pd_multiindex(multi)
print(f" Type = {type(mi)},shape {mi.shape}")
