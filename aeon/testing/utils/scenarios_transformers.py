"""Test scenarios for transformers.

Contains TestScenario concrete children to run in tests for transformers.
"""

__maintainer__ = []

__all__ = ["scenarios_transformers"]

import inspect
from copy import deepcopy
from inspect import isclass

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from aeon.base import BaseObject
from aeon.testing.utils.data_gen import (
    _make_classification_y,
    _make_collection_X,
    _make_hierarchical,
    make_series,
)
from aeon.testing.utils.scenarios import TestScenario
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.validation import abstract_types

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42
RAND_SEED2 = 84

# typical length of time series
N_T = 10


def _make_primitives(n_columns=1, random_state=None):
    """Generate one or more primitives, for checking inverse-transform."""
    rng = check_random_state(random_state)
    if n_columns == 1:
        return rng.rand()
    return rng.rand(size=(n_columns,))


def _make_tabular_X(n_cases=20, n_columns=1, return_numpy=True, random_state=None):
    """Generate tabular X, for checking inverse-transform."""
    rng = check_random_state(random_state)
    X = rng.rand(n_cases, n_columns)
    if return_numpy:
        return X
    else:
        return pd.DataFrame(X)


def _is_child_of(obj, class_or_tuple):
    """Shorthand for 'inherits from', obj can be class or object."""
    if isclass(obj):
        return issubclass(obj, class_or_tuple)
    else:
        return isinstance(obj, class_or_tuple)


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


def _internal_abstract_type(obj, inner_tag, series_types):
    inner_types = get_tag(obj, inner_tag)
    if isinstance(inner_types, str):
        inner_types = {inner_types}
    else:
        inner_types = set(inner_types)
    return not inner_types.issubset(series_types)


class TransformerTestScenario(TestScenario, BaseObject):
    """Generic test scenario for transformers."""

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """
        # pre-refactor classes can't deal with Series *and* Panel both
        X_type = self.get_tag("X_type")
        y_type = self.get_tag("y_type", None, raise_error=False)

        if (
            isinstance(obj, BaseCollectionTransformer)
            or (inspect.isclass(obj) and issubclass(obj, BaseCollectionTransformer))
        ) and X_type != "Panel":
            return False

        # if transformer requires y, the scenario also must pass y
        has_y = self.get_tag("has_y")
        if not has_y and get_tag(obj, "requires_y"):
            return False

        # the case that we would need to vectorize with y, skip
        X_inner_type = get_tag(obj, "X_inner_type")
        X_inner_abstract_types = abstract_types(X_inner_type)
        # we require vectorization from of a Series trafo to Panel data ...
        if X_type == "Panel" and "Panel" not in X_inner_abstract_types:
            # ... but y is passed and y is not ignored internally ...
            if self.get_tag("has_y") and get_tag(obj, "y_inner_type") != "None":
                # ... this would raise an error since vectorization is not defined
                return False

        # ensure scenario y matches type of inner y
        y_inner_type = get_tag(obj, "y_inner_type")
        if y_inner_type not in [None, "None"]:
            y_inner_abstract_types = abstract_types(y_inner_type)
            if y_type not in y_inner_abstract_types:
                return False

        # only applicable if X of supported index type
        X = self.args["fit"]["X"]
        supported_idx_types = get_tag(obj, "enforce_index_type")
        if isinstance(X, (pd.Series, pd.DataFrame)) and supported_idx_types is not None:
            if type(X.index) not in supported_idx_types:
                return False
        if isinstance(X, np.ndarray) and supported_idx_types is not None:
            if pd.RangeIndex not in supported_idx_types:
                return False

        return True

    def get_args(self, key, obj=None, deepcopy_args=False):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        if key == "inverse_transform":
            if obj is None:
                raise ValueError('if key="inverse_transform", obj must be provided')

            X_type = self.get_tag("X_type")

            X_out_type = get_tag(obj, "output_data_type")
            X_panel = get_tag(obj, "instancewise")

            X_out_series = X_out_type == "Series"
            X_out_prim = X_out_type == "Primitives"

            # determine output by X_type
            s2s = X_type == "Series" and X_out_series
            s2p = X_type == "Series" and X_out_prim
            p2t = X_type == "Panel" and X_out_prim
            p2p = X_type == "Panel" and X_out_series

            # expected input type of inverse_transform is expected output of transform
            if s2p:
                args = {"X": _make_primitives(random_state=RAND_SEED)}
            elif s2s:
                args = {"X": make_series(n_timepoints=N_T, random_state=RAND_SEED)}
            elif p2t:
                args = {"X": _make_tabular_X(n_cases=7, nrandom_state=RAND_SEED)}
            elif p2p:
                args = {
                    "X": _make_collection_X(
                        n_cases=7, n_timepoints=N_T, random_state=RAND_SEED
                    )
                }
            else:
                raise RuntimeError(
                    "transformer with unexpected combination of tags: "
                    f"X_out_type = {X_out_type}, instancewise = {X_panel}"
                )

        else:
            # default behaviour, happens except when key = "inverse_transform"
            args = self.args.get(key, {})

        if deepcopy_args:
            args = deepcopy(args)

        return args


X_series = make_series(n_timepoints=N_T, random_state=RAND_SEED)
X_panel = _make_collection_X(
    n_cases=7, n_channels=1, n_timepoints=N_T, random_state=RAND_SEED
)


class TransformerFitTransformSeriesUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Series X."""

    _tags = {
        # These tags are only used in testing and are not defined in the registry
        "X_type": "Series",
        "X_univariate": True,
        "has_y": False,
        "is_enabled": True,
    }

    args = {
        "fit": {"X": make_series(n_timepoints=N_T + 1, random_state=RAND_SEED)},
        "transform": {"X": make_series(n_timepoints=N_T + 1, random_state=RAND_SEED2)},
        # "inverse_transform": {"X": make_series(n_timepoints=N_T)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformSeriesMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Series X."""

    _tags = {
        "X_type": "Series",
        "X_univariate": False,
        "has_y": False,
        "is_enabled": True,
    }

    args = {
        "fit": {
            "X": make_series(n_columns=2, n_timepoints=N_T, random_state=RAND_SEED),
        },
        "transform": {
            "X": make_series(n_columns=2, n_timepoints=N_T, random_state=RAND_SEED)
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformSeriesUnivariateWithY(TransformerTestScenario):
    """Fit/transform, univariate Series X and univariate Series y."""

    _tags = {
        "X_type": "Series",
        "X_univariate": True,
        "has_y": True,
        "is_enabled": True,
        "y_type": "Series",
    }

    args = {
        "fit": {
            "X": make_series(n_columns=1, n_timepoints=N_T, random_state=RAND_SEED),
            "y": make_series(n_columns=1, n_timepoints=N_T, random_state=RAND_SEED),
        },
        "transform": {
            "X": make_series(n_columns=1, n_timepoints=N_T, random_state=RAND_SEED),
            "y": make_series(n_columns=1, n_timepoints=N_T, random_state=RAND_SEED),
        },
    }
    default_method_sequence = ["fit", "transform"]


y3 = _make_classification_y(n_cases=9, n_classes=3)
X_np = _make_collection_X(
    n_cases=9,
    n_channels=1,
    n_timepoints=N_T,
    all_positive=True,
    return_numpy=True,
    random_state=RAND_SEED,
)
X_test_np = _make_collection_X(
    n_cases=9,
    n_channels=1,
    n_timepoints=N_T,
    all_positive=True,
    return_numpy=True,
    random_state=RAND_SEED2,
)


class TransformerFitTransformPanelUnivariateNumpyWithClassYOnlyFit(
    TransformerTestScenario
):
    """Fit/predict with univariate panel X, numpy3D input type, and labels y."""

    _tags = {
        "X_type": "Panel",
        "X_univariate": True,
        "has_y": True,
        "is_enabled": True,
        "y_type": "Table",
    }

    args = {
        "fit": {"y": y3, "X": X_np},
        "transform": {"X": X_test_np},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Panel X."""

    _tags = {
        "X_type": "Panel",
        "X_univariate": True,
        "has_y": False,
        "is_enabled": False,
    }

    args = {
        "fit": {
            "X": _make_collection_X(
                n_cases=7, n_channels=1, n_timepoints=N_T, random_state=RAND_SEED
            )
        },
        "transform": {
            "X": _make_collection_X(
                n_cases=7, n_channels=1, n_timepoints=N_T, random_state=RAND_SEED
            )
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Panel X."""

    _tags = {
        "X_type": "Panel",
        "X_univariate": False,
        "has_y": False,
        "is_enabled": False,
    }

    args = {
        "fit": {
            "X": _make_collection_X(
                n_cases=7, n_channels=2, n_timepoints=N_T, random_state=RAND_SEED
            )
        },
        "transform": {
            "X": _make_collection_X(
                n_cases=7, n_channels=2, n_timepoints=N_T, random_state=RAND_SEED
            )
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariateWithClassY(TransformerTestScenario):
    """Fit/transform, multivariate Panel X, with y in fit and transform."""

    _tags = {
        "X_type": "Panel",
        "X_univariate": True,
        "is_enabled": True,
        "has_y": True,
        "y_type": "Table",
    }

    args = {
        "fit": {
            "X": _make_collection_X(
                n_cases=7,
                n_channels=1,
                n_timepoints=N_T + 1,
                all_positive=True,
                random_state=RAND_SEED,
            ),
            "y": _make_classification_y(n_cases=7, n_classes=2),
        },
        "transform": {
            "X": _make_collection_X(
                n_cases=7,
                n_channels=1,
                n_timepoints=N_T + 1,
                all_positive=True,
                random_state=RAND_SEED,
            ),
            "y": _make_classification_y(n_cases=7, n_classes=2),
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariateWithClassYOnlyFit(TransformerTestScenario):
    """Fit/transform, multivariate Panel X, with y in fit but not in transform."""

    _tags = {
        "X_type": "Panel",
        "X_univariate": True,
        "is_enabled": False,
        "has_y": True,
        "y_type": "Table",
    }

    args = {
        "fit": {
            "X": _make_collection_X(n_cases=7, n_channels=1, n_timepoints=N_T),
            "y": _make_classification_y(n_cases=7, n_classes=2),
        },
        "transform": {
            "X": _make_collection_X(n_cases=7, n_channels=1, n_timepoints=N_T)
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformHierarchicalUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Hierarchical X."""

    _tags = {
        "X_type": "Hierarchical",
        "X_univariate": True,
        "is_enabled": False,
        "has_y": False,
    }

    args = {
        "fit": {"X": _make_hierarchical(random_state=RAND_SEED)},
        "transform": {"X": _make_hierarchical(random_state=RAND_SEED + 1)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformHierarchicalMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Hierarchical X."""

    _tags = {
        "X_type": "Hierarchical",
        "X_univariate": False,
        "is_enabled": False,
        "has_y": False,
    }

    args = {
        "fit": {"X": _make_hierarchical(random_state=RAND_SEED, n_columns=2)},
        "transform": {"X": _make_hierarchical(random_state=RAND_SEED + 1, n_columns=2)},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers = [
    TransformerFitTransformSeriesUnivariate,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformSeriesUnivariateWithY,
    TransformerFitTransformPanelUnivariate,
    TransformerFitTransformPanelMultivariate,
    TransformerFitTransformPanelUnivariateWithClassY,
    TransformerFitTransformPanelUnivariateWithClassYOnlyFit,
    TransformerFitTransformPanelUnivariateNumpyWithClassYOnlyFit,
    TransformerFitTransformHierarchicalMultivariate,
    TransformerFitTransformHierarchicalUnivariate,
]
