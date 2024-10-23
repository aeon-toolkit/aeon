"""Test expected outputs for estimators."""

import numpy as np
import pytest

from aeon.testing.expected_results.expected_classifier_outputs import (
    basic_motions_proba,
    unit_test_proba,
)
from aeon.testing.expected_results.expected_regressor_outputs import (
    cardano_sentiment_preds,
    covid_3month_preds,
)
from aeon.testing.expected_results.expected_transform_outputs import (
    basic_motions_result,
    unit_test_result,
)
from aeon.testing.testing_config import PR_TESTING
from aeon.utils.discovery import all_estimators


@pytest.mark.skipif(
    PR_TESTING,
    reason="Don't want to run all_estimators multiple times every PR.",
)
def test_expected_classifier_outputs():
    """Test estimators in the expected classifier outputs dict."""
    classifiers = all_estimators(type_filter=["classifier", "early_classifier"])
    classifier_names = [c[0] for c in classifiers]

    for key, value in unit_test_proba.items():
        assert key in classifier_names
        assert isinstance(value, np.ndarray)

    for key, value in basic_motions_proba.items():
        assert key in classifier_names
        assert isinstance(value, np.ndarray)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Don't want to run all_estimators multiple times every PR.",
)
def test_expected_regressor_outputs():
    """Test estimators in the expected regressor outputs dict."""
    regressors = all_estimators(type_filter="regressor")
    regressor_names = [r[0] for r in regressors]

    for key, value in covid_3month_preds.items():
        assert key in regressor_names
        assert isinstance(value, np.ndarray)

    for key, value in cardano_sentiment_preds.items():
        assert key in regressor_names
        assert isinstance(value, np.ndarray)


@pytest.mark.skipif(
    PR_TESTING,
    reason="Don't want to run all_estimators multiple times every PR.",
)
def test_expected_transformer_outputs():
    """Test estimators in the expected transformer outputs dict."""
    transformers = all_estimators(type_filter="transformer")
    transformer_names = [r[0] for r in transformers]

    for key, value in unit_test_result.items():
        assert key in transformer_names
        assert isinstance(value, np.ndarray)

    for key, value in basic_motions_result.items():
        assert key in transformer_names
        assert isinstance(value, np.ndarray)
