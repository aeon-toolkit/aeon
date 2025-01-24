"""Test function for SMOTE."""

import numpy as np
import pytest

from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.imbalance import SMOTE
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies(
        "imbalanced-learn",
        package_import_alias={"imbalanced-learn": "imblearn"},
        severity="none",
    ),
    reason="skip test if required soft dependency imbalanced-learn not available",
)
def test_equivalence_imbalance():
    """Test ported SMOTE code produces the same as imblearn version."""
    from imblearn.over_sampling import SMOTE as imbSMOTE

    X, y = make_example_3d_numpy(n_cases=20, n_channels=1)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    X = X.squeeze()
    s1 = imbSMOTE(random_state=49)
    X2, y2 = s1.fit_resample(X, y)
    s2 = SMOTE(random_state=49)
    X3, y3 = s2.fit_transform(X, y)
    X3 = X3.squeeze()
    assert np.array_equal(y2, y3)
    assert np.allclose(X2, X3, atol=1e-4)
