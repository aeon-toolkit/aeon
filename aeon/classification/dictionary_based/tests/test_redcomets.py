"""REDCOMETS test code."""

import numpy as np
import pytest
from sklearn.utils import check_random_state

from aeon.classification.dictionary_based import REDCOMETS

N_PER_CLASS = 10
N_TIMEPOINTS = 48  # long enough to yield >=2 SFA and >=2 SAX lenses per view
N_CLASSES = 2


def _labelled_panel(n_channels, n_per_class=N_PER_CLASS, random_state=0):
    """Return a balanced random panel with ``N_CLASSES`` classes.

    Values are random: the tests assert output structure (shape, valid labels,
    normalised probabilities), not classification accuracy.
    """
    rng = check_random_state(random_state)
    n_cases = n_per_class * N_CLASSES
    X = rng.standard_normal((n_cases, n_channels, N_TIMEPOINTS))
    y = np.repeat(np.arange(N_CLASSES), n_per_class)
    return X, y


def _assert_valid_output(clf, X):
    """Check output structure: normalised probabilities and in-vocabulary labels."""
    proba = clf.predict_proba(X)
    pred = clf.predict(X)

    assert proba.shape == (X.shape[0], clf.n_classes_)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert pred.shape == (X.shape[0],)
    assert set(pred).issubset(set(clf.classes_))


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_redcomets_univariate_variants(variant):
    """Univariate variants 1-3 fit and produce well-formed predictions."""
    X, y = _labelled_panel(n_channels=1)
    clf = REDCOMETS(variant=variant, n_trees=3, random_state=0)
    clf.fit(X, y)
    _assert_valid_output(clf, X)


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_redcomets_multivariate_concatenate_variants(variant):
    """Variants 1-3 handle multivariate input by concatenating channels."""
    X, y = _labelled_panel(n_channels=3)
    clf = REDCOMETS(variant=variant, n_trees=3, random_state=0)
    clf.fit(X, y)
    _assert_valid_output(clf, X)


@pytest.mark.parametrize("variant", [4, 5, 6, 7, 8, 9])
def test_redcomets_dimension_ensemble_variants(variant):
    """Variants 4-9 build and fuse a per-channel ensemble on multivariate input.

    These variants exercise the dimension-ensemble build and the variant-specific
    fusion in ``_predict_proba_dimension_ensemble`` (plain sum vs. confidence
    weighting, at both the per-channel and cross-channel stages).
    """
    X, y = _labelled_panel(n_channels=3)
    clf = REDCOMETS(variant=variant, n_trees=3, random_state=0)
    clf.fit(X, y)
    _assert_valid_output(clf, X)


def test_redcomets_deterministic():
    """A fixed random_state gives identical predictions across fits."""
    X, y = _labelled_panel(n_channels=3)

    pred1 = REDCOMETS(variant=5, n_trees=3, random_state=0).fit(X, y).predict(X)
    pred2 = REDCOMETS(variant=5, n_trees=3, random_state=0).fit(X, y).predict(X)

    np.testing.assert_array_equal(pred1, pred2)


def test_redcomets_balanced_input_needs_no_oversampling():
    """Already-balanced classes fit without invoking the oversampling branch."""
    X, y = _labelled_panel(n_channels=1)  # N_PER_CLASS each, balanced
    assert np.unique(y, return_counts=True)[1].tolist() == [N_PER_CLASS, N_PER_CLASS]

    clf = REDCOMETS(variant=1, n_trees=3, random_state=0)
    clf.fit(X, y)
    _assert_valid_output(clf, X)


def test_redcomets_imbalanced_input_uses_smote():
    """An imbalanced class large enough for neighbour search is SMOTE-oversampled.

    The minority class has more than five samples, exercising the capped
    neighbour-count SMOTE path rather than the fallback.
    """
    X, _ = _labelled_panel(n_channels=1, n_per_class=14)
    y = np.array([0] * 20 + [1] * 8)  # minority > 5 -> capped SMOTE neighbours

    clf = REDCOMETS(variant=1, n_trees=3, random_state=0)
    clf.fit(X, y)
    assert set(clf.classes_) == {0, 1}
    _assert_valid_output(clf, X)


def test_redcomets_tiny_minority_uses_random_oversampler():
    """A minority class too small for SMOTE falls back to random oversampling.

    With two minority samples the SMOTE neighbour count drops below one, so
    REDCOMETS must fall back to RandomOverSampler and still fit on both classes.
    """
    X, _ = _labelled_panel(n_channels=1, n_per_class=10)
    y = np.array([0] * 18 + [1] * 2)

    clf = REDCOMETS(variant=1, n_trees=3, random_state=0)
    clf.fit(X, y)
    assert set(clf.classes_) == {0, 1}
    _assert_valid_output(clf, X)


@pytest.mark.parametrize("bad_variant", [0, 10])
def test_redcomets_rejects_invalid_variant(bad_variant):
    """Variants outside 1-9 are rejected at construction."""
    with pytest.raises(AssertionError):
        REDCOMETS(variant=bad_variant)


@pytest.mark.parametrize("bad_perc", [0, 101])
def test_redcomets_rejects_invalid_perc_length(bad_perc):
    """perc_length must lie in (0, 100]."""
    with pytest.raises(AssertionError):
        REDCOMETS(perc_length=bad_perc)


def test_redcomets_univariate_rejects_ensemble_variant():
    """Dimension-ensemble variants 4-9 require multivariate input."""
    X, y = _labelled_panel(n_channels=1)
    clf = REDCOMETS(variant=4, n_trees=3, random_state=0)
    with pytest.raises(AssertionError):
        clf.fit(X, y)


def test_redcomets_test_params_are_valid():
    """The documented test parameters construct a valid REDCOMETS instance."""
    params = REDCOMETS._get_test_params()
    assert params["variant"] in range(1, 10)
    REDCOMETS(**params)  # construction asserts pass


def test_redcomets_declares_no_imbalanced_learn_dependency():
    """REDCOMETS no longer depends on imbalanced-learn (gh-3654)."""
    deps = REDCOMETS(random_state=0).get_tag("python_dependencies", None)
    if deps is None:
        deps = []
    if isinstance(deps, str):
        deps = [deps]
    assert "imblearn" not in deps
    assert "imbalanced-learn" not in deps
