"""Test composable estimator mixin."""

import pytest

from aeon.classification.compose import ClassifierEnsemble
from aeon.testing.mock_estimators import MockClassifier, MockClassifierParams
from aeon.testing.testing_data import EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION


def test_get_params():
    """Tst get_params retrieval for composable estimators."""
    ens = [("clf1", MockClassifierParams()), ("clf2", MockClassifierParams())]
    clf = ClassifierEnsemble(ens)

    params = clf.get_params(deep=False)

    expected = {
        "classifiers",
        "cv",
        "majority_vote",
        "metric",
        "metric_probas",
        "random_state",
        "weights",
    }

    assert isinstance(params, dict)
    assert set(params.keys()) == expected
    assert params["classifiers"] == ens

    params = clf.get_params()

    expected = expected.union(
        {
            "clf1",
            "clf2",
            "clf1__return_ones",
            "clf1__value",
            "clf2__return_ones",
            "clf2__value",
        }
    )

    assert isinstance(params, dict)
    assert set(params.keys()) == expected
    assert params["clf1__value"] == 50


def test_set_params():
    """Test set_params for composable estimators."""
    clf = ClassifierEnsemble(
        [("clf1", MockClassifierParams()), ("clf2", MockClassifierParams())]
    )

    ens = [("clf3", MockClassifierParams()), ("clf4", MockClassifierParams())]
    params = {"_ensemble": ens, "clf3__value": 100, "clf4__return_ones": True}
    clf.set_params(**params)

    assert clf._ensemble[0][1].value == 100
    assert clf._ensemble[1][1].return_ones is True


def test_get_fitted_params():
    """Test get_fitted_params for composable estimators."""
    X, y = EQUAL_LENGTH_UNIVARIATE_CLASSIFICATION["numpy3D"]["train"]

    clf = ClassifierEnsemble(
        [("clf1", MockClassifierParams()), ("clf2", MockClassifierParams())]
    )
    clf.fit(X, y)

    params = clf.get_fitted_params(deep=False)

    expected = {
        "classes_",
        "ensemble_",
        "metadata_",
        "n_classes_",
        "weights_",
    }

    assert isinstance(params, dict)
    assert set(params.keys()) == expected
    assert params["n_classes_"] == clf.n_classes_

    params = clf.get_fitted_params()

    expected = expected.union(
        {
            "clf1",
            "clf1__classes_",
            "clf1__foo_",
            "clf1__metadata_",
            "clf1__n_classes_",
            "clf2",
            "clf2__classes_",
            "clf2__foo_",
            "clf2__metadata_",
            "clf2__n_classes_",
        }
    )

    assert isinstance(params, dict)
    assert set(params.keys()) == expected
    assert params["clf1__n_classes_"] == 2


def test_check_estimators():
    """Test check_estimators for composable estimators."""
    ens = [("clf1", MockClassifier()), MockClassifier()]
    clf = ClassifierEnsemble(ens)

    clf._check_estimators(ens, unique_names=False)

    with pytest.raises(ValueError, match="estimators should only contain singular"):
        clf._check_estimators(ens, allow_tuples=False)

    with pytest.raises(ValueError, match="should only contain"):
        clf._check_estimators(ens, allow_single_estimators=False)

    with pytest.raises(ValueError, match="must be an instance of"):
        clf._check_estimators([("class", MockClassifier)])

    with pytest.raises(ValueError, match="must be of form"):
        clf._check_estimators([(MockClassifier(),)])

    with pytest.raises(ValueError, match="must be of form"):
        clf._check_estimators([(MockClassifier, "class")])

    with pytest.raises(ValueError, match="conflicts with constructor arguments"):
        clf._check_estimators([("classifiers", MockClassifier())])

    with pytest.raises(ValueError, match="Estimator name must not contain"):
        clf._check_estimators([("__clf", MockClassifier())])

    with pytest.raises(ValueError, match="must be unique"):
        clf._check_estimators(
            [("clf", MockClassifier()), ("clf", MockClassifier())], unique_names=True
        )

    with pytest.raises(ValueError, match="name is invalid"):
        clf._check_estimators(ens, invalid_names=["clf1"])

    with pytest.raises(ValueError, match="name is invalid"):
        clf._check_estimators(ens, invalid_names="clf1")

    with pytest.raises(TypeError, match="tuple or estimator"):
        clf._check_estimators(["invalid"])

    with pytest.raises(TypeError, match="Invalid estimators attribute"):
        clf._check_estimators([])


def test_convert_estimators():
    """Test convert_estimators for composable estimators."""
    ens = [
        ("clf1", MockClassifierParams()),
        MockClassifierParams(),
        MockClassifierParams(),
    ]
    clf = ClassifierEnsemble(ens)
    ens2 = clf._convert_estimators(ens)

    assert isinstance(ens2, list)
    assert len(ens2) == 3
    assert ens2[0][0] == "clf1"
    assert ens2[1][0] == "MockClassifierParams_0"
    assert ens2[2][0] == "MockClassifierParams_1"
    assert isinstance(ens2[0][1], MockClassifierParams)
    assert isinstance(ens2[1][1], MockClassifierParams)
    assert isinstance(ens2[2][1], MockClassifierParams)
