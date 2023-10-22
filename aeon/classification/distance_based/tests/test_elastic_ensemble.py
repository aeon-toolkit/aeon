"""Tests for ElasticEnsemble."""
import numpy as np
import pytest

from aeon.classification.distance_based import ElasticEnsemble

DISTANCE = [
    "lcss",
    "dtw",
    "wdtw",
    "erp",
    "msm",
]
PARAS = {
    "dtw": {"window"},
    "wdtw": {"g"},
    "lcss": {"epsilon", "window"},
    "erp": {"g", "window"},
    "msm": {"c"},
}
DATA = np.random.random((10, 1, 50))


@pytest.mark.parametrize("dist", DISTANCE)
@pytest.mark.parametrize("data", [None, DATA])
def test_get_100_param_options(dist, data):
    """Test the method to get 100 options per distance function.
    1. Test 100 returned.
    2. Test on specified range.
    """
    if (
        dist == "erp" or dist == "lcss"
    ) and data is None:  # raise exception, LCSS needs
        # train data
        with pytest.raises(ValueError):
            ElasticEnsemble._get_100_param_options(dist, data)
    else:
        paras = ElasticEnsemble._get_100_param_options(dist, data)
        para_values = paras["distance_params"]
        # Check correct number of para combos
        assert len(para_values) == 100
        # Check all provided parameters are valid for this distance
        expected_paras = PARAS[dist]
        for p in para_values:
            for ep in expected_paras:
                assert ep in p


def test_proportion_train_in_param_finding():
    """Test proportion in train for parameter finding."""
    X = np.random.random(size=(10, 1, 10))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    ee = ElasticEnsemble(
        distance_measures=["dtw"], proportion_train_in_param_finding=0.1
    )
    with pytest.raises(
        ValueError, match="should be greater or equal to the number of classes"
    ):
        ee.fit(X, y)
    ee = ElasticEnsemble(
        distance_measures=["ddtw", "wddtw"],
        proportion_train_in_param_finding=0.2,
        verbose=True,
    )
    ee.fit(X, y)
    ee.get_metric_params()
    with pytest.raises(NotImplementedError, match="EE does not currently support:"):
        ElasticEnsemble._get_100_param_options("FOO")
