"""Tests for the CLeVer family of channel selectors."""

import numpy as np
import pytest

from aeon.testing.estimator_checking import check_estimator
from aeon.transformations.collection.channel_selection import (
    CLeVerCluster,
    CLeVerHybrid,
    CLeVerRank,
)
from aeon.transformations.collection.channel_selection._clever import (
    _case_correlation_eigensystem,
    _stable_descending_order,
)
from aeon.transformations.collection.channel_selection.base import BaseChannelSelector

CLEVER_CLASSES = [CLeVerRank, CLeVerCluster, CLeVerHybrid]


def _make_clever_data(seed=0, n_cases=12, n_timepoints=30):
    """Create correlated channels with a nondegenerate eigenspectrum."""
    rng = np.random.default_rng(seed)
    X = np.empty((n_cases, 6, n_timepoints))
    for case in range(n_cases):
        first = rng.normal(size=n_timepoints)
        second = rng.normal(size=n_timepoints)
        third = rng.normal(size=n_timepoints)
        X[case, 0] = first + 0.02 * rng.normal(size=n_timepoints)
        X[case, 1] = first + 0.25 * rng.normal(size=n_timepoints)
        X[case, 2] = 0.65 * first + 0.35 * second
        X[case, 3] = second + 0.03 * rng.normal(size=n_timepoints)
        X[case, 4] = second + 0.4 * rng.normal(size=n_timepoints)
        X[case, 5] = third + 0.1 * first
    return X


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_standard_estimator_checks(estimator_class):
    """Run the applicable standard aeon estimator checks."""
    check_estimator(
        estimator_class,
        raise_exceptions=True,
        use_first_parameter_set=True,
    )


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_inheritance_unsupervised_and_shape(estimator_class):
    """CLeVer selectors inherit correctly, ignore y, and select exactly K channels."""
    X = _make_clever_data()
    kwargs = {"n_channels": 3}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 7
    first = estimator_class(**kwargs)
    second = estimator_class(**kwargs)

    first.fit(X, np.arange(len(X)) % 2)
    second.fit(X, np.arange(len(X)) % 3)

    assert isinstance(first, BaseChannelSelector)
    assert first.get_tag("requires_y") is False
    assert first.channels_selected_ == second.channels_selected_
    assert len(first.channels_selected_) == 3
    assert len(set(first.channels_selected_)) == 3
    assert all(0 <= channel < X.shape[1] for channel in first.channels_selected_)
    assert first.channels_selected_ == sorted(first.channels_selected_)
    assert first.transform(X).shape == (X.shape[0], 3, X.shape[2])
    np.testing.assert_allclose(first.common_matrix_, second.common_matrix_)
    np.testing.assert_allclose(
        first.dcpc_.T @ first.dcpc_, second.dcpc_.T @ second.dcpc_
    )
    if estimator_class is not CLeVerRank:
        np.testing.assert_array_equal(first.cluster_labels_, second.cluster_labels_)


def test_clever_controlled_dcpc_projection_and_orientation():
    """Check the shared DCPC stage against an independently derived subspace."""
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    X = np.array(
        [
            [signal, signal, -signal],
            [3 * signal, 0.5 * signal, -2 * signal],
        ]
    )
    selector = CLeVerRank(n_channels=1, variance_threshold=0.8).fit(X)

    direction = np.array([1.0, 1.0, -1.0]) / np.sqrt(3.0)
    expected_projection = np.outer(direction, direction)
    np.testing.assert_array_equal(selector.case_n_components_, [1, 1])
    assert selector.n_components_ == 1
    assert selector.dcpc_.shape == (1, 3)
    np.testing.assert_allclose(
        selector.dcpc_.T @ selector.dcpc_, expected_projection, atol=1e-12
    )
    np.testing.assert_allclose(
        selector.common_matrix_, 2 * expected_projection, atol=1e-12
    )


def test_clever_uses_maximum_case_component_count():
    """Use the maximum per-case count when constructing every case projection."""
    first = np.array([1.0, 1.0, -1.0, -1.0])
    second = np.array([1.0, -1.0, 1.0, -1.0])
    third = np.array([1.0, -1.0, -1.0, 1.0])
    X = np.array(
        [
            [first, first, first],
            [first, second, third],
        ]
    )
    selector = CLeVerRank(n_channels=1, variance_threshold=0.8).fit(X)

    np.testing.assert_array_equal(selector.case_n_components_, [1, 3])
    assert selector.n_components_ == 3
    np.testing.assert_allclose(selector.common_matrix_, 2 * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(selector.dcpc_.T @ selector.dcpc_, np.eye(3), atol=1e-12)


def test_clever_rank_order_and_exact_ties():
    """Rank in descending order and resolve exact score ties by channel index."""
    scores = np.array([0.5, 0.8, 0.8, 0.2, 0.5])
    np.testing.assert_array_equal(_stable_descending_order(scores), [1, 2, 0, 4, 3])

    X = _make_clever_data()
    selector = CLeVerRank(n_channels=3).fit(X)
    expected = np.lexsort((np.arange(X.shape[1]), -selector.channel_scores_))
    np.testing.assert_array_equal(selector.channel_ranking_, expected)
    assert set(selector.channels_selected_) == set(expected[:3])


def test_clever_cluster_selects_nearest_cluster_members():
    """CLeVer-Cluster chooses the nearest original member of every centroid."""
    selector = CLeVerCluster(n_channels=3, random_state=4).fit(_make_clever_data())
    expected = []
    for cluster_index, center in enumerate(selector.cluster_centers_):
        members = np.flatnonzero(selector.cluster_labels_ == cluster_index)
        distances = np.linalg.norm(selector.channel_loadings_[members] - center, axis=1)
        expected.append(int(members[np.lexsort((members, distances))[0]]))

    assert selector.channels_selected_ == sorted(expected)
    assert selector.inertia_ >= 0


def test_clever_hybrid_selects_highest_scored_cluster_members():
    """CLeVer-Hybrid chooses the highest CLeVer-Rank score in every cluster."""
    selector = CLeVerHybrid(n_channels=3, random_state=4).fit(_make_clever_data())
    expected = []
    for cluster_index in range(selector.n_channels):
        members = np.flatnonzero(selector.cluster_labels_ == cluster_index)
        order = np.lexsort((members, -selector.channel_scores_[members]))
        expected.append(int(members[order[0]]))

    assert selector.channels_selected_ == sorted(expected)


def test_clever_cluster_and_hybrid_share_clustering():
    """Cluster and Hybrid reuse equivalent clustering with identical parameters."""
    X = _make_clever_data(seed=1)
    cluster = CLeVerCluster(n_channels=3, random_state=19).fit(X)
    hybrid = CLeVerHybrid(n_channels=3, random_state=19).fit(X)

    np.testing.assert_array_equal(cluster.cluster_labels_, hybrid.cluster_labels_)
    np.testing.assert_allclose(cluster.cluster_centers_, hybrid.cluster_centers_)
    assert cluster.inertia_ == pytest.approx(hybrid.inertia_)
    assert len(cluster.channels_selected_) == len(hybrid.channels_selected_) == 3
    assert cluster.channels_selected_ != hybrid.channels_selected_


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_fixed_seed_repeated_fits(estimator_class):
    """Repeated fits are deterministic when the random state is fixed."""
    X = _make_clever_data()
    kwargs = {"n_channels": 3}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 11
    selector = estimator_class(**kwargs)
    selector.fit(X)
    first_selected = selector.channels_selected_.copy()
    first_projection = selector.dcpc_.T @ selector.dcpc_
    first_labels = getattr(selector, "cluster_labels_", None)
    if first_labels is not None:
        first_labels = first_labels.copy()

    selector.fit(X)

    assert selector.channels_selected_ == first_selected
    np.testing.assert_allclose(selector.dcpc_.T @ selector.dcpc_, first_projection)
    if first_labels is not None:
        np.testing.assert_array_equal(selector.cluster_labels_, first_labels)


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_case_order_invariance(estimator_class):
    """Reordering cases preserves the common subspace and selected channels."""
    X = _make_clever_data()
    kwargs = {"n_channels": 3}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 5
    first = estimator_class(**kwargs).fit(X)
    second = estimator_class(**kwargs).fit(X[::-1])

    np.testing.assert_allclose(first.common_matrix_, second.common_matrix_, atol=1e-12)
    np.testing.assert_allclose(
        first.dcpc_.T @ first.dcpc_,
        second.dcpc_.T @ second.dcpc_,
        atol=1e-12,
    )
    assert first.channels_selected_ == second.channels_selected_


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_constant_channel_in_one_case(estimator_class):
    """A channel constant in one case remains usable through the other cases."""
    X = _make_clever_data()
    X[0, 2] = 4.0
    kwargs = {"n_channels": 2}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 0
    selector = estimator_class(**kwargs).fit(X)
    assert len(selector.channels_selected_) == 2
    assert np.all(np.isfinite(selector.common_matrix_))


def test_clever_constant_channel_has_zero_case_correlation_projection():
    """A constant channel contributes a zero row and column within its case."""
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    X_case = np.array([signal, np.ones(5), -signal])
    eigenvectors, n_components = _case_correlation_eigensystem(
        X_case, case_index=0, variance_threshold=0.8
    )

    direction = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    expected_projection = np.outer(direction, direction)
    actual_projection = (
        eigenvectors[:, :n_components] @ eigenvectors[:, :n_components].T
    )

    assert n_components == 1
    np.testing.assert_allclose(actual_projection, expected_projection, atol=1e-12)


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_all_channels_constant_in_case(estimator_class):
    """An entirely constant case has no explained variance and raises clearly."""
    X = _make_clever_data()
    X[3] = 1.0
    kwargs = {"n_channels": 2}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 0
    with pytest.raises(ValueError, match="All channels in case 3 are constant"):
        estimator_class(**kwargs).fit(X)


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_one_channel_and_select_all(estimator_class):
    """One-channel inputs and K equal to all channels satisfy the selector contract."""
    rng = np.random.default_rng(0)
    X_one = rng.normal(size=(5, 1, 10))
    kwargs = {"n_channels": 1}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 0
    one = estimator_class(**kwargs).fit(X_one)
    assert one.channels_selected_ == [0]
    assert one.transform(X_one).shape == X_one.shape

    X = _make_clever_data(n_cases=8)
    kwargs["n_channels"] = X.shape[1]
    all_channels = estimator_class(**kwargs).fit(X)
    assert all_channels.channels_selected_ == list(range(X.shape[1]))


@pytest.mark.parametrize("estimator_class", CLEVER_CLASSES)
def test_clever_two_timepoints(estimator_class):
    """Very short nonconstant series remain valid correlation inputs."""
    X = np.array(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[2.0, 4.0], [-1.0, 1.0]],
            [[-2.0, 3.0], [4.0, -3.0]],
        ]
    )
    kwargs = {"n_channels": 1}
    if estimator_class is not CLeVerRank:
        kwargs["random_state"] = 0
    selector = estimator_class(**kwargs).fit(X)
    assert len(selector.channels_selected_) == 1


def test_clever_rank_is_cumulative_in_k():
    """Increasing K in CLeVer-Rank adds the next ranked channel."""
    X = _make_clever_data()
    selected = [
        set(CLeVerRank(n_channels=k).fit(X).channels_selected_) for k in (1, 2, 3)
    ]
    assert selected[0] < selected[1] < selected[2]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_channels": 0}, "n_channels"),
        ({"n_channels": 7}, "n_channels"),
        ({"n_channels": 1.5}, "n_channels"),
        ({"n_channels": True}, "n_channels"),
        ({"variance_threshold": 0}, "variance_threshold"),
        ({"variance_threshold": 1.1}, "variance_threshold"),
        ({"variance_threshold": np.nan}, "variance_threshold"),
        ({"variance_threshold": "0.8"}, "variance_threshold"),
    ],
)
def test_clever_common_parameter_validation(kwargs, match):
    """Common invalid parameters are rejected."""
    with pytest.raises(ValueError, match=match):
        CLeVerRank(**kwargs).fit(_make_clever_data(n_cases=4))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_init": 0}, "n_init"),
        ({"n_init": 1.0}, "n_init"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": True}, "max_iter"),
        ({"tol": -1e-4}, "tol"),
        ({"tol": np.inf}, "tol"),
        ({"tol": "1e-4"}, "tol"),
    ],
)
def test_clever_clustering_parameter_validation(kwargs, match):
    """Invalid clustering parameters are rejected."""
    with pytest.raises(ValueError, match=match):
        CLeVerCluster(n_channels=2, **kwargs).fit(_make_clever_data(n_cases=4))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_clever_rejects_nonfinite_input(value):
    """Non-finite input is rejected before numerical decomposition."""
    X = _make_clever_data(n_cases=4)
    X[0, 0, 0] = value
    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError):
            CLeVerRank(n_channels=2).fit(X)


def test_clever_duplicate_loadings_raise_for_empty_clusters():
    """Too few distinct loading vectors produce a clear exact-K contract error."""
    selector = CLeVerCluster(n_channels=2, random_state=0)
    selector.dcpc_ = np.ones((1, 3))
    with pytest.raises(ValueError, match="fewer non-empty clusters"):
        selector._cluster_channels()
