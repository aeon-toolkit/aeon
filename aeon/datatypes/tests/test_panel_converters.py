"""Testing collection converters - internal functions and more extensive fixtures."""

import numpy as np
import pandas as pd
import pytest

from aeon.datatypes._adapter import convert_from_multiindex_to_listdataset
from aeon.datatypes._panel._check import (
    are_columns_nested,
    check_nplist_panel,
    is_nested_dataframe,
)
from aeon.datatypes._panel._convert import (
    from_2d_array_to_nested,
    from_3d_numpy_to_2d_array,
    from_3d_numpy_to_multi_index,
    from_3d_numpy_to_nested,
    from_3d_numpy_to_nplist,
    from_dflist_to_nplist,
    from_long_to_nested,
    from_multi_index_to_3d_numpy,
    from_multi_index_to_nested,
    from_nested_to_2d_array,
    from_nested_to_3d_numpy,
    from_nested_to_long,
    from_nested_to_multi_index,
    from_nested_to_nplist,
    from_nplist_to_nested,
    from_numpy3d_to_dflist,
)
from aeon.testing.utils.data_gen import (
    make_example_long_table,
    make_example_multi_index_dataframe,
    make_example_nested_dataframe,
    make_example_unequal_length,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies

N_CASES = [10, 15]
N_CHANNELS = [3, 5]
N_TIMEPOINTS = [3, 5]
N_CLASSES = [2, 5]


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_are_columns_nested(n_cases, n_channels, n_timepoints):
    """Test are_columns_nested for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    zero_df = pd.DataFrame(np.zeros_like(nested))
    nested_heterogenous1 = pd.concat([zero_df, nested], axis=1)
    nested_heterogenous2 = nested.copy()
    nested_heterogenous2["primitive_col"] = 1.0

    assert [*are_columns_nested(nested)] == [True] * n_channels
    assert [*are_columns_nested(nested_heterogenous1)] == [False] * n_channels + [
        True
    ] * n_channels
    assert [*are_columns_nested(nested_heterogenous2)] == [True] * n_channels + [False]


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_3d_numpy(n_cases, n_channels, n_timepoints):
    """Test from_nested_to_3d_numpy for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    array = from_nested_to_3d_numpy(nested)

    # check types and shapes
    assert isinstance(array, np.ndarray)
    assert array.shape == (n_cases, n_channels, n_timepoints)

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], array[1, 0, :])


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_nested(n_cases, n_channels, n_timepoints):
    """Test from_3d_numpy_to_nested for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))
    nested = from_3d_numpy_to_nested(array)

    # check types and shapes
    assert is_nested_dataframe(nested)
    assert nested.shape == (n_cases, n_channels)
    assert nested.iloc[0, 0].shape[0] == n_timepoints

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], array[1, 0, :])


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_2d_array(n_cases, n_channels, n_timepoints):
    """Test from_nested_to_2d_array for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)

    array = from_nested_to_2d_array(nested)
    assert array.shape == (n_cases, n_channels * n_timepoints)
    assert array.index.equals(nested.index)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_2d_array(n_cases, n_channels, n_timepoints):
    """Test from_3d_numpy_to_2d_array for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))
    array_2d = from_3d_numpy_to_2d_array(array)

    assert array_2d.shape == (n_cases, n_channels * n_timepoints)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multi_index_to_3d_numpy(n_cases, n_channels, n_timepoints):
    """Test from_multi_index_to_3d_numpy for correctness."""
    mi_df = make_example_multi_index_dataframe(
        n_cases=n_cases, n_timepoints=n_timepoints, n_channels=n_channels
    )

    array = from_multi_index_to_3d_numpy(mi_df)

    assert isinstance(array, np.ndarray)
    assert array.shape == (n_cases, n_channels, n_timepoints)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_multi_index(n_cases, n_channels, n_timepoints):
    """Test from_3d_numpy_to_multi_index for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))

    mi_df = from_3d_numpy_to_multi_index(
        array, instance_index=None, time_index=None, column_names=None
    )

    col_names = ["column_" + str(i) for i in range(n_channels)]
    mi_df_named = from_3d_numpy_to_multi_index(
        array, instance_index="case_id", time_index="reading_id", column_names=col_names
    )

    assert isinstance(mi_df, pd.DataFrame)
    assert mi_df.index.names == ["instances", "timepoints"]
    assert (mi_df.columns == ["var_" + str(i) for i in range(n_channels)]).all()

    assert isinstance(mi_df_named, pd.DataFrame)
    assert mi_df_named.index.names == ["case_id", "reading_id"]
    assert (mi_df_named.columns == col_names).all()


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multi_index_to_nested(n_cases, n_channels, n_timepoints):
    """Test from_multi_index_to_nested for correctness."""
    mi_df = make_example_multi_index_dataframe(
        n_cases=n_cases, n_timepoints=n_timepoints, n_channels=n_channels
    )
    nested_df = from_multi_index_to_nested(
        mi_df, instance_index="case_id", cells_as_numpy=False
    )

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_cases, n_channels)
    assert (nested_df.columns == mi_df.columns).all()


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_multi_index(n_cases, n_channels, n_timepoints):
    """Test from_nested_to_multi_index for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    mi_df = from_nested_to_multi_index(
        nested, instance_index="case_id", time_index="reading_id"
    )

    # n_timepoints_max = nested.applymap(_nested_cell_timepoints).sum().max()

    assert isinstance(mi_df, pd.DataFrame)
    assert mi_df.shape == (n_cases * n_timepoints, n_channels)
    assert mi_df.index.names == ["case_id", "reading_id"]


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_is_nested_dataframe(n_cases, n_channels, n_timepoints):
    """Test is_nested_dataframe for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    zero_df = pd.DataFrame(np.zeros_like(nested))
    nested_heterogenous = pd.concat([zero_df, nested], axis=1)

    mi_df = make_example_multi_index_dataframe(
        n_cases=n_cases, n_timepoints=n_timepoints, n_channels=n_channels
    )

    assert not is_nested_dataframe(array)
    assert not is_nested_dataframe(mi_df)
    assert is_nested_dataframe(nested)
    assert is_nested_dataframe(nested_heterogenous)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_2d_array_to_nested(n_cases, n_channels, n_timepoints):
    """Test from_2d_array_to_nested for correctness."""
    rng = np.random.default_rng()
    X_2d = rng.standard_normal((n_cases, n_timepoints))
    nested_df = from_2d_array_to_nested(X_2d)

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_cases, 1)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_long_to_nested(n_cases, n_channels, n_timepoints):
    """Test from_long_to_nested for correctness."""
    X_long = make_example_long_table(
        n_cases=n_cases, n_timepoints=n_timepoints, n_channels=n_channels
    )
    nested_df = from_long_to_nested(X_long)

    assert is_nested_dataframe(nested_df)
    assert nested_df.shape == (n_cases, n_channels)


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_long(n_cases, n_channels, n_timepoints):
    """Test from_nested_to_long for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    X_long = from_nested_to_long(
        nested,
        instance_column_name="case_id",
        time_column_name="reading_id",
        dimension_column_name="dim_id",
    )

    assert isinstance(X_long, pd.DataFrame)
    assert X_long.shape == (n_cases * n_timepoints * n_channels, 4)
    assert (X_long.columns == ["case_id", "reading_id", "dim_id", "value"]).all()


@pytest.mark.skipif(
    not _check_soft_dependencies("gluonts", severity="none"),
    reason="requires gluonts package in the example",
)
@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_multiindex_to_listdataset(n_cases, n_channels, n_timepoints):
    """Test from multiindex DF to listdataset for gluonts."""
    import numpy as np
    import pandas as pd

    from aeon.datatypes import convert_to

    # from aeon.datatypes._adapters import convert_from_multiindex_to_listdataset

    def random_datetimes_or_dates(
        start, end, out_format="datetime", n=10, random_seed=42
    ):
        """Generate random pd Datetime in the start to end range.

        unix timestamp is in ns by default.
        Divide the unix time value by 10**9 to make it seconds
        (or 24*60*60*10**9 to make it days).
        The corresponding unit variable is passed to the pd.to_datetime function.
        Values for the (divide_by, unit) pair to select is defined by the out_format
        parameter.
        for 1 -> out_format='datetime'
        for 2 -> out_format=anything else.
        """
        np.random.seed(random_seed)
        (divide_by, unit) = (
            (10**9, "s") if out_format == "datetime" else (24 * 60 * 60 * 10**9, "D")
        )

        start_u = start.value // divide_by
        end_u = end.value // divide_by

        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit)

    def _make_example_multiindex(
        n_cases, n_channels, n_timepoints, random_seed=42
    ) -> pd.DataFrame:
        import numpy as np

        start = pd.to_datetime("1750-01-01")
        end = pd.to_datetime("2022-07-01")
        inputDF = np.random.randint(1, 99, size=(n_cases * n_timepoints, n_channels))
        n_cases = n_cases
        column_name = []
        for i in range(n_channels):
            column_name.append("dim_" + str(i))

        random_start_date = random_datetimes_or_dates(
            start, end, out_format="out datetime", n=n_cases, random_seed=42
        )

        level0_idx = [
            list(np.full(n_timepoints, instance)) for instance in range(n_cases)
        ]
        level0_idx = np.ravel(level0_idx)

        level1_idx = [
            list(
                pd.date_range(
                    random_start_date[instance], periods=n_timepoints, freq="H"
                )
            )
            for instance in range(n_cases)
        ]
        level1_idx = np.ravel(level1_idx)

        multi_idx = pd.MultiIndex.from_arrays(
            [level0_idx, level1_idx], names=("instance", "datetime")
        )

        inputDF_return = pd.DataFrame(inputDF, columns=column_name, index=multi_idx)

        return inputDF_return

    MULTIINDEX_DF = _make_example_multiindex(n_cases, n_channels, n_timepoints)
    # Result from the converter
    listdataset_result = convert_from_multiindex_to_listdataset(MULTIINDEX_DF)
    listdataset_result_list = list(listdataset_result)
    # Result from raw data
    dimension_name = MULTIINDEX_DF.columns
    # Convert MULTIINDEX_DF to nested_univ format to compare with listdataset
    control_result = convert_to(MULTIINDEX_DF, to_type="nested_univ")
    control_result = control_result.reset_index()
    control_result = control_result[dimension_name]

    # Perform the test
    for instance, _dim_name in control_result.iterrows():
        for dim_no, dim in enumerate(dimension_name):
            np.testing.assert_array_equal(
                control_result.loc[instance, dim].to_numpy(),
                listdataset_result_list[instance]["target"][dim_no],
            )


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_3d_numpy_to_nplist(n_cases, n_channels, n_timepoints):
    """Test from_3d_numpy_to_nplist for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))
    np_list = from_3d_numpy_to_nplist(array)

    # check types and shapes
    correct, _ = check_nplist_panel(np_list)
    assert correct
    assert len(np_list) == n_cases
    assert np_list[0].shape == (n_channels, n_timepoints)

    # check values of random series
    np.testing.assert_array_equal(np_list[0], array[0])


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_dflist_to_nplist(n_cases, n_channels, n_timepoints):
    """Test from_dflist_to_nplist for correctness."""
    array = np.random.normal(size=(n_cases, n_channels, n_timepoints))
    df_list = from_numpy3d_to_dflist(array)
    np_list = from_dflist_to_nplist(df_list)

    # check types and shapes
    correct, _ = check_nplist_panel(np_list)
    assert correct
    assert len(np_list) == n_cases
    assert np_list[0].shape == (n_channels, n_timepoints)

    # check values of random series
    np.testing.assert_array_equal(df_list[0], array[0].transpose())


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nested_to_nplist(n_cases, n_channels, n_timepoints):
    """Test from_nested_to_nplist for correctness."""
    nested, _ = make_example_nested_dataframe(n_cases, n_channels, n_timepoints)
    np_list = from_nested_to_nplist(nested)

    # check types and shapes
    correct, _ = check_nplist_panel(np_list)
    assert correct
    assert len(np_list) == n_cases
    assert np_list[0].shape == (n_channels, n_timepoints)

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], np_list[1][0])


@pytest.mark.parametrize("n_cases", N_CASES)
@pytest.mark.parametrize("n_channels", N_CHANNELS)
@pytest.mark.parametrize("n_timepoints", N_TIMEPOINTS)
def test_from_nplist_to_nested(n_cases, n_channels, n_timepoints):
    """Test from_nplist_to_nested for correctness."""
    np_list, _ = make_example_unequal_length(
        n_cases, n_channels, n_timepoints, n_timepoints
    )
    nested = from_nplist_to_nested(np_list)

    # check types and shapes
    assert is_nested_dataframe(nested)
    assert nested.shape == (n_cases, n_channels)
    assert nested.iloc[0, 0].shape[0] == n_timepoints

    # check values of random series
    np.testing.assert_array_equal(nested.iloc[1, 0], np_list[1][0])
