"""Test summary classifier."""

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from aeon.classification.feature_based import TSFreshClassifier
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_tsfresh_classifier():
    """Test the TSFreshClassifier."""
    X, y = make_example_3d_numpy()
    cls = TSFreshClassifier(estimator=RandomForestClassifier(n_jobs=1))
    cls.fit(X, y)
    p = cls.predict_proba(X)
    assert cls.estimator_.n_jobs == 1 and cls.n_jobs == 1
    cls = TSFreshClassifier(estimator=RidgeClassifier())
    cls.fit(X, y)
    p = cls.predict_proba(X)
    assert np.all(np.isin(p, [0, 1]))
    with warnings.catch_warnings():
        X, y = make_example_3d_numpy(
            n_cases=2, n_timepoints=3, random_state=0, n_labels=2
        )
        cls = TSFreshClassifier(relevant_feature_extractor=True)
        cls.fit(X, y)
        assert cls._return_majority_class is True
        assert cls._majority_class in [0, 1]
    cls.verbose = 1
    cls.fit(X, y)
