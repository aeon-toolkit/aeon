"""Unit tests for base class conversion and vectorization functionality.

Each test covers a "decision path" in the base class boilerplate,
    with a focus on frequently breaking paths in base class refactor and bugfixing.
The path taken depends on tags of a given transformer, and input data type.
Concrete transformer classes from aeon are imported to cover
    different combinations of transformer tags.
Transformer scenarios cover different combinations of input data types.
"""

__maintainer__ = []
__all__ = []

from inspect import isclass

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from aeon.datatypes import mtype_to_scitype
from aeon.testing.utils.data_gen import get_examples, make_series
from aeon.testing.utils.scenarios_transformers import (
    TransformerFitTransformHierarchicalMultivariate,
    TransformerFitTransformHierarchicalUnivariate,
    TransformerFitTransformPanelUnivariate,
    TransformerFitTransformPanelUnivariateWithClassYOnlyFit,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformSeriesUnivariate,
)
from aeon.transformations.base import BaseTransformer
from aeon.transformations.boxcox import BoxCoxTransformer
from aeon.transformations.compose import FitInTransform
from aeon.utils.validation import (
    is_collection,
    is_hierarchical,
    is_single_series,
    is_tabular,
)


def inner_X_types(est):
    """Return list of types supported by class est, as list of str."""
    if isclass(est):
        X_inner_type = est.get_class_tag("X_inner_type")
    else:
        X_inner_type = est.get_tag("X_inner_type")
    X_inner_types = mtype_to_scitype(
        X_inner_type, return_unique=True, coerce_to_list=True
    )
    return X_inner_types


class _DummyOne(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "numpy3D",
        "fit_is_empty": False,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X


def test_series_in_series_out_not_supported_but_panel():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" does not support "Series" but does support "Panel"

    X input to fit/transform is a Series
    X output from fit/transform should be Series
    """
    cls = _DummyOne
    est = cls.create_test_instance()
    assert "Panel" in inner_X_types(est)
    assert "Series" not in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])

    assert is_single_series(Xt), (
        "fit.transform does not return a Series when given a " "Series"
    )


def test_collection_in_collection_out_supported():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Panel"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Panel
    """
    cls = _DummyOne
    est = cls.create_test_instance()
    assert "Panel" in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_collection(Xt), (
        "fit.transform does not return a collection when given " "a collection"
    )


class _DummyTwo(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "np.ndarray",
        "univariate-only": True,
        "fit_is_empty": False,
    }

    def _transform(self, X, y=None):
        return X


def test_series_in_series_out_supported():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Series

    X input to fit/transform has Series scitype
    X ouput from fit/transform should be Series
    """
    cls = _DummyTwo
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_single_series(
        Xt
    ), "fit.transform does not return a Series when given a Series"


def test_collection_in_collection_out_not_supported_but_series():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Panel
    """
    cls = _DummyTwo
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_collection(Xt), (
        "fit.transform does not return a collection when given " "a collection"
    )


def test_broadcast_multivariate_no_row_broadcast_empty_fit():
    """Test that multivariate vectorization of univariate transformers works.

    This test should trigger column (variable) vectorization, but not row vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = True
        "X_inner_type" supports "Series"

    X input to fit/transform has Series scitype, is multivariate
    X output from fit/transform should be Series and multivariate
    """
    # one example for a transformer which supports Series internally
    cls = _DummyTwo
    est = FitInTransform(cls.create_test_instance())
    assert "Series" in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    assert est.get_tag("univariate-only")
    scenario = TransformerFitTransformSeriesMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_single_series(
        Xt
    ), "fit.transform does not return a series when given a series"
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


def test_hierarchical_in_hierarchical_out_not_supported_but_series():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    cls = _DummyTwo
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformHierarchicalUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_hierarchical(Xt), (
        "fit.transform does not return a Hierarchical when " "given Hierarchical"
    )
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == 2 * 4 * 12


def test_vectorization_multivariate_and_hierarchical():
    """Test that fit/transform runs and returns the correct output type.

    This test should trigger both column (variable) and row (hierarchy) vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Series" but not "Panel" and not "Hierarchical

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    cls = _DummyTwo
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    assert est.get_tag("univariate-only")
    scenario = TransformerFitTransformHierarchicalMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_hierarchical(Xt), (
        "fit.transform does not return a Hierarchical when " "given Hierarchical"
    )
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


def test_vectorization_multivariate_and_hierarchical_empty_fit():
    """Test that fit/transform runs and returns the correct output type.

    This test should trigger both column (variable) and row (hierarchy) vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = True
        "X_inner_type" supports "Series" but not "Panel" and not "Hierarchical

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    cls = _DummyTwo
    est = FitInTransform(cls.create_test_instance())
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    assert est.get_tag("univariate-only")
    scenario = TransformerFitTransformHierarchicalMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_hierarchical(Xt), (
        "fit.transform does not return a Hierarchical when " "given Hierarchical"
    )
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


class _DummyThree(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Series",
        "X_inner_type": "np.ndarray",
        "fit_is_empty": True,
    }

    def _transform(self, X, y=None):
        return X


def test_series_in_series_out_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = True
        "X_inner_type" supports "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Series
    """
    cls = _DummyThree
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert est.get_class_tag("fit_is_empty")
    assert est.get_class_tag("input_data_type") == "Series"
    assert est.get_class_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_single_series(
        Xt
    ), "fit.transform does not return a series when given a series"


def test_hierarchical_in_hierarchical_out_not_supported_but_series_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = True
        "X_inner_type" supports "Series" but not "Panel" and not "Hierarchical"

    X input to fit/transform has Hierarchical scitype
    X output from fit/transform should be Hierarchical
    """
    cls = _DummyThree
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    scenario = TransformerFitTransformHierarchicalUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_hierarchical(Xt), (
        "fit.transform does not return a Hierarchical when " "given Hierarchical"
    )
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == 2 * 4 * 12


class _DummyFour(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Primitives",
        "X_inner_type": ["pd.DataFrame"],
        "y_inner_type": "None",
        "fit_is_empty": True,
    }

    def _transform(self, X, y=None):
        return np.array([0.0])


def test_series_in_primitives_out_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Primitives"
        "fit_is_empty" = True
        "X_inner_type" supports "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Table
    """
    cls = _DummyFour
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Primitives"
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    # length of Xt should be one, for a single series passed
    assert len(Xt) == 1
    assert is_tabular(Xt), "fit.transform does not return a Table when given a Series"


def test_panel_in_primitives_out_not_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Primitives"
        "fit_is_empty" = True
        "X_inner_type" does not support "Panel", but does supports "Series"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Table
    """
    cls = _DummyFour
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert "Panel" not in inner_X_types(est)
    assert "Hierarchical" not in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Primitives"
    scenario = TransformerFitTransformPanelUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_tabular(Xt), "fit.transform does not return a Table when given a Panel"
    # length of Xt should be seven = number of samples in the scenario
    assert len(Xt) == 7


def test_vectorize_reconstruct_correct_hierarchy():
    """Tests correct transform return index in hierarchical case for primitives output.

    Tests that the row index is as expected if rows are vectorized over,
    by a transform that returns Primitives.
    The row index of transform return should be identical to the input,
    with temporal index level removed

    Raises
    ------
    AssertionError if output index is not as expected.
    """
    from aeon.testing.utils.data_gen import _make_hierarchical

    # hierarchical data with 2 variables and 2 levels
    X = _make_hierarchical(n_columns=2)

    trafo = _DummyFour()

    # this produces a pandas DataFrame with more rows and columns
    # rows should correspond to different instances in X
    Xt = trafo.fit_transform(X)

    # check that Xt.index is the same as X.index with time level dropped and made unique
    assert (X.index.droplevel(-1).unique() == Xt.index).all()


class _DummyFive(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Primitives",
        "X_inner_type": ["numpy3D"],
        "fit_is_empty": True,
    }

    def _transform(self, X, y=None):
        return np.array([0.0])


def test_series_in_primitives_out_not_supported_fit_in_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Primitives"
        "fit_is_empty" = True
        "X_inner_type" supports "Panel" but does not support "Series"

    X input to fit/transform has Series scitype
    X output from fit/transform should be Table
    """
    cls = _DummyFive
    est = cls.create_test_instance()
    assert "Panel" in inner_X_types(est)
    assert "Series" not in inner_X_types(est)
    assert est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Primitives"
    scenario = TransformerFitTransformSeriesUnivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_tabular(Xt), "fit.transform does not return a Table when given a Series"
    # length of Xt should be one, for a single series passed
    assert len(Xt) == 1


class _DummySix(BaseTransformer):
    _tags = {
        "input_data_type": "Series",
        "output_data_type": "Primitives",
        "X_inner_type": ["numpy3D"],
        "fit_is_empty": False,
        "requires_y": True,
    }

    def _transform(self, X, y=None):
        return np.zeros(len(X))


def test_panel_in_primitives_out_supported_with_y_in_fit_but_not_transform():
    """Test that fit/transform runs and returns the correct output type.

    Setting: transformer has tags
        "input_data_type" = "Series"
        "output_data_type" = "Primitives"
        "fit_is_empty" = False
        "requires_y" = True
        "X_inner_type" supports "Panel"

    X input to fit/transform has Panel scitype
    X output from fit/transform should be Table
    """
    cls = _DummySix
    est = cls.create_test_instance()
    assert "Panel" in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("requires_y")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Primitives"
    scenario = TransformerFitTransformPanelUnivariateWithClassYOnlyFit()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_tabular(Xt), "fit.transform does not return a Table when given a Panel"
    # length of Xt should be seven = number of samples in the scenario
    assert len(Xt) == 7


def test_vectorization_multivariate_no_row_vectorization():
    """Test that multivariate vectorization of univariate transformers works.

    This test should trigger column (variable) vectorization, but not row vectorization.

    Setting: transformer has tags
        "univariate-only" = True
        "input_data_type" = "Series"
        "output_data_type" = "Series"
        "fit_is_empty" = False
        "X_inner_type" supports "Series"

    X input to fit/transform has Series scitype, is multivariate
    X output from fit/transform should be Series and multivariate
    """
    cls = _DummyTwo
    est = cls.create_test_instance()
    assert "Series" in inner_X_types(est)
    assert not est.get_tag("fit_is_empty")
    assert est.get_tag("input_data_type") == "Series"
    assert est.get_tag("output_data_type") == "Series"
    assert est.get_tag("univariate-only")
    scenario = TransformerFitTransformSeriesMultivariate()
    Xt = scenario.run(est, method_sequence=["fit", "transform"])
    assert is_single_series(
        Xt
    ), "fit.transform does not return a Series when given a Series"
    # length of Xt should be number of hierarchy levels times number of time points
    assert len(Xt) == len(scenario.args["fit"]["X"])
    assert len(Xt.columns) == len(scenario.args["fit"]["X"].columns)


def test_vectorize_reconstruct_unique_columns():
    """Tests that vectorization on multivariate output yields unique columns.

    Also test that the column names are as expected:
    <variable>__<transformed> if multiple transformed variables per variable are present
    <variable> if one variable is transformed to one output

    Raises
    ------
    AssertionError if output columns are not as expected.
    """
    from aeon.transformations.detrend import Detrender
    from aeon.transformations.theta import ThetaLinesTransformer

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    X_mi = get_examples("pd_multiindex_hier")[0]
    t = ThetaLinesTransformer()
    X_t_cols = t.fit_transform(X).columns
    assert set(X_t_cols) == {"a__0", "a__2", "b__0", "b__2", "c__0", "c__2"}
    X_mi_cols = t.fit_transform(X_mi)
    assert set(X_mi_cols) == {"var_0__0", "var_0__2", "var_1__0", "var_1__2"}
    X = make_series(n_columns=2, n_timepoints=15)
    t = Detrender.create_test_instance()
    Xt = t.fit_transform(X)
    assert set(Xt.columns) == {0, 1}


def test_numpy_format_outputs():
    """Test that all numpy formats return the same output when converted."""
    X = np.random.random(size=(2, 1, 8))
    bc = BoxCoxTransformer()

    u1d = bc.fit_transform(X[0][0])
    # 2d numpy arrays are (n_timepoints, n_channels) while 3d numpy arrays are
    # (n_cases, n_channels, n_timepoints)
    u2d = bc.fit_transform(X[0].transpose()).transpose()
    u3d = bc.fit_transform(X)

    assert_array_equal(u1d, u2d[0])
    assert_array_equal(u1d, u3d[0][0])

    X = np.random.random(size=(2, 2, 8))

    m2d = bc.fit_transform(X[0].transpose()).transpose()
    m3d = bc.fit_transform(X)

    assert_array_equal(m2d, m3d[0])
