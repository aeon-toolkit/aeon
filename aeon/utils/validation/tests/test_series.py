"""Test series validation module."""

__maintainer__ = ["TonyBagnall", "MatthewMiddlehurst"]


from aeon.testing.data_generation import (
    make_example_1d_numpy,
    make_example_2d_numpy_series,
    make_example_3d_numpy,
    make_example_3d_numpy_list,
    make_example_dataframe_series,
    make_example_multi_index_dataframe,
    make_example_pandas_series,
)
from aeon.utils.validation.series import (
    is_series,
    is_univariate,
)


def test_is_series():
    """Test is_series validation."""
    assert not is_series(None)
    assert is_series(make_example_pandas_series())
    assert is_series(make_example_1d_numpy())
    assert is_series(make_example_dataframe_series(n_channels=1))
    assert is_series(make_example_dataframe_series(n_channels=2))
    assert is_series(make_example_2d_numpy_series())
    assert not is_series(make_example_3d_numpy())
    assert not is_series(make_example_3d_numpy_list())


def test_is_univariate():
    """Test is_univariate validation."""
    # pandas
    assert is_univariate(make_example_pandas_series())

    # numpy
    assert is_univariate(make_example_1d_numpy())
    assert is_univariate(make_example_dataframe_series(n_channels=1))
    assert not is_univariate(make_example_dataframe_series(n_channels=2))
    assert not is_univariate(make_example_2d_numpy_series())

    # collections
    assert not is_univariate(make_example_3d_numpy())
    assert not is_univariate(make_example_3d_numpy_list())
    make_example_multi_index_dataframe()

    # other
    assert not is_univariate(None)
    assert not is_univariate("str")
