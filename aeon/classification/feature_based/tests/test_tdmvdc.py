"""Test TDMVDC Classifier."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from aeon.classification.feature_based import TDMVDCClassifier
from aeon.datasets import (
    load_arrow_head,
    load_classification,
    load_gunpoint,
    load_italy_power_demand,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
def test_tdmvdc_classifier():
    """Test the TDMVDCClassifier."""
    cls = TDMVDCClassifier()
    assert cls.k1 == 2 and cls.k2 == 2
    assert cls.n_jobs == 1


@pytest.mark.skipif(
    not _check_soft_dependencies("tsfresh", severity="none"),
    reason="skip test if required soft dependency tsfresh not available",
)
@pytest.mark.parametrize(
    "dataset_name,expected_accuracy",
    [
        ("ArrowHead", 0.8114),
        ("Beef", 0.9667),
        ("BeetleFly", 0.95),
        ("GunPoint", 0.993),
        ("ItalyPowerDemand", 0.965),
    ],
)
def check_tdmvdc_results(dataset_name, expected_accuracy):
    """Check the results of TDMVDCClassifier with expected accuracy."""
    # Load the dataset
    if dataset_name == "ArrowHead":
        trainSignalX, trainY = load_arrow_head("TRAIN")
        testSignalX, testY = load_arrow_head("TEST")
    elif dataset_name == "GunPoint":
        trainSignalX, trainY = load_gunpoint("TRAIN")
        testSignalX, testY = load_gunpoint("TEST")
    elif dataset_name == "ItalyPowerDemand":
        trainSignalX, trainY = load_italy_power_demand("TRAIN")
        testSignalX, testY = load_italy_power_demand("TEST")
    else:
        trainSignalX, trainY = load_classification(dataset_name, "TRAIN")
        testSignalX, testY = load_classification(dataset_name, "TEST")
    trainY, testY = trainY.astype(int), testY.astype(int)

    cls = TDMVDCClassifier(n_jobs=2)
    cls.fit(trainSignalX, trainY)
    preds = cls.predict(testSignalX)
    acc = accuracy_score(testY, preds)
    assert np.isclose(
        acc, expected_accuracy, atol=0.02
    ), f"Accuracy {acc:.2f} not close to expected \
    {expected_accuracy:.2f} for {dataset_name}"
