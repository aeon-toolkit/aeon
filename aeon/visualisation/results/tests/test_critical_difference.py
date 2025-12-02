"""Tests for the critical difference diagram maker."""

import os

import numpy as np
import pytest

import aeon
from aeon.benchmarking.results_loaders import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation.results._critical_difference import (
    _build_cliques,
    _compute_adjusted_alpha,
    _compute_ranks,
    _rank_to_position,
    _sort_by_average_rank,
    plot_critical_difference,
)

data_path = os.path.join(
    os.path.dirname(aeon.__file__),
    "testing/example_results_files/",
)

test_clique1 = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique1 = np.array(
    [
        [False, False, True, True],
    ]
)
test_clique2 = np.array(
    [
        [True, False, False, False],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique2 = np.array([[False, False, True, True]])

test_clique3 = np.array(
    [
        [True, True, True, True, True, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False, False, False, False],
        [False, False, True, True, True, True, False, False, False, False, False],
        [False, False, False, True, True, True, True, True, True, True, True],
        [False, False, False, False, True, True, True, True, True, True, False],
        [False, False, False, False, False, True, True, True, True, True, True],
        [False, False, False, False, False, False, True, True, True, True, False],
        [False, False, False, False, False, False, False, True, True, True, True],
        [False, False, False, False, False, False, False, False, True, True, True],
        [False, False, False, False, False, False, False, False, False, True, True],
        [False, False, False, False, False, False, False, False, False, False, True],
    ]
)
correct_clique3 = np.array(
    [
        [True, True, True, True, True, False, False, False, False, False, False],
        [False, True, True, True, True, True, True, False, False, False, False],
        [False, False, False, True, True, True, True, True, True, True, True],
    ]
)
test_clique4 = np.array(
    [
        [True, True, True, True],
        [False, True, False, False],
        [False, False, True, True],
        [False, False, False, True],
    ]
)
correct_clique4 = np.array([[True, True, True, True]])


CLIQUES = [
    (test_clique1, correct_clique1),
    (test_clique2, correct_clique2),
    (test_clique3, correct_clique3),
    (test_clique4, correct_clique4),
]


@pytest.mark.parametrize("cliques", CLIQUES)
def test__build_cliques(cliques):
    """Test build cliques with different cases."""
    obs_clique = _build_cliques(cliques[0])
    assert np.all(obs_clique == cliques[1])


def test__build_cliques_empty():
    """Test build cliques with empty return."""
    test_clique = np.array([[True, False], [False, True]])
    obs_clique = _build_cliques(test_clique)
    assert obs_clique.size == 0


# =============================================================================
# Tests for _compute_ranks
# =============================================================================


def test__compute_ranks_higher_better():
    """Test rank computation when higher scores are better."""
    scores = np.array([[0.9, 0.8, 0.7], [0.6, 0.7, 0.8]])
    ranks = _compute_ranks(scores, lower_better=False)
    # Higher scores should get lower (better) ranks
    assert ranks[0, 0] == 1  # 0.9 is best in row 0
    assert ranks[0, 2] == 3  # 0.7 is worst in row 0
    assert ranks[1, 2] == 1  # 0.8 is best in row 1


def test__compute_ranks_lower_better():
    """Test rank computation when lower scores are better."""
    scores = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    ranks = _compute_ranks(scores, lower_better=True)
    # Lower scores should get lower (better) ranks
    assert ranks[0, 0] == 1  # 0.1 is best in row 0
    assert ranks[0, 2] == 3  # 0.3 is worst in row 0
    assert ranks[1, 2] == 1  # 0.1 is best in row 1


def test__compute_ranks_ties():
    """Test rank computation with ties (average ranks assigned)."""
    scores = np.array([[0.9, 0.9, 0.7]])
    ranks = _compute_ranks(scores, lower_better=False)
    # Tied scores should get average rank
    assert ranks[0, 0] == 1.5  # Tied for 1st
    assert ranks[0, 1] == 1.5  # Tied for 1st
    assert ranks[0, 2] == 3    # 3rd place


# =============================================================================
# Tests for _sort_by_average_rank
# =============================================================================


def test__sort_by_average_rank():
    """Test sorting estimators by average rank."""
    scores = np.array([[0.9, 0.7, 0.8], [0.8, 0.6, 0.9]])
    labels = ["A", "B", "C"]
    ranks = _compute_ranks(scores, lower_better=False)

    ordered_labels, ordered_avg_ranks, ordered_scores = _sort_by_average_rank(
        scores, labels, ranks
    )

    # Check labels are sorted by average rank
    assert ordered_labels[0] in ["A", "C"]  # Best performers
    assert ordered_labels[-1] == "B"  # Worst performer

    # Check ranks are in ascending order
    assert np.all(np.diff(ordered_avg_ranks) >= 0)


# =============================================================================
# Tests for _compute_adjusted_alpha
# =============================================================================


def test__compute_adjusted_alpha_bonferroni():
    """Test Bonferroni correction."""
    alpha = 0.1
    n_estimators = 5
    # Bonferroni: alpha / (n * (n-1) / 2) = 0.1 / 10 = 0.01
    adjusted = _compute_adjusted_alpha(alpha, n_estimators, "bonferroni")
    assert adjusted == pytest.approx(0.01)


def test__compute_adjusted_alpha_holm():
    """Test Holm correction."""
    alpha = 0.1
    n_estimators = 5
    # Holm: alpha / (n - 1) = 0.1 / 4 = 0.025
    adjusted = _compute_adjusted_alpha(alpha, n_estimators, "holm")
    assert adjusted == pytest.approx(0.025)


def test__compute_adjusted_alpha_none():
    """Test no correction."""
    alpha = 0.1
    n_estimators = 5
    adjusted = _compute_adjusted_alpha(alpha, n_estimators, None)
    assert adjusted == alpha


def test__compute_adjusted_alpha_invalid():
    """Test invalid correction raises error."""
    with pytest.raises(ValueError, match="correction available"):
        _compute_adjusted_alpha(0.1, 5, "invalid")


# =============================================================================
# Tests for _rank_to_position
# =============================================================================


def test__rank_to_position_normal():
    """Test rank to position conversion without reverse."""
    textspace = 1.0
    scalewidth = 4.0
    min_rank = 1
    max_rank = 5

    # Rank 1 should be at textspace
    pos = _rank_to_position(1, textspace, scalewidth, min_rank, max_rank, reverse=False)
    assert pos == pytest.approx(1.0)

    # Rank 5 should be at textspace + scalewidth
    pos = _rank_to_position(5, textspace, scalewidth, min_rank, max_rank, reverse=False)
    assert pos == pytest.approx(5.0)

    # Rank 3 should be in the middle
    pos = _rank_to_position(3, textspace, scalewidth, min_rank, max_rank, reverse=False)
    assert pos == pytest.approx(3.0)


def test__rank_to_position_reversed():
    """Test rank to position conversion with reverse."""
    textspace = 1.0
    scalewidth = 4.0
    min_rank = 1
    max_rank = 5

    # With reverse, rank 1 should be at textspace + scalewidth
    pos = _rank_to_position(1, textspace, scalewidth, min_rank, max_rank, reverse=True)
    assert pos == pytest.approx(5.0)

    # With reverse, rank 5 should be at textspace
    pos = _rank_to_position(5, textspace, scalewidth, min_rank, max_rank, reverse=True)
    assert pos == pytest.approx(1.0)


# =============================================================================
# Tests for plot_critical_difference input validation
# =============================================================================


def test_plot_critical_difference_invalid_scores_shape():
    """Test that invalid score shape raises error."""
    scores = np.array([0.9, 0.8, 0.7])  # 1D instead of 2D
    labels = ["A", "B", "C"]
    with pytest.raises(ValueError, match="must be a 2D array"):
        plot_critical_difference(scores, labels)


def test_plot_critical_difference_mismatched_labels():
    """Test that mismatched label count raises error."""
    scores = np.array([[0.9, 0.8, 0.7]])
    labels = ["A", "B"]  # Only 2 labels for 3 estimators
    with pytest.raises(ValueError, match="Number of labels"):
        plot_critical_difference(scores, labels)


def test_plot_critical_difference_too_few_estimators():
    """Test that too few estimators raises error."""
    scores = np.array([[0.9]])  # Only 1 estimator
    labels = ["A"]
    with pytest.raises(ValueError, match="at least 2 estimators"):
        plot_critical_difference(scores, labels)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_critical_difference_p_values_nemenyi_error():
    """Test that requesting p_values with nemenyi raises error."""
    scores = np.array([[0.9, 0.8], [0.7, 0.6]])
    labels = ["A", "B"]
    with pytest.raises(ValueError, match="Cannot return p values"):
        plot_critical_difference(scores, labels, test="nemenyi", return_p_values=True)


# =============================================================================
# Tests for plot_critical_difference
# =============================================================================


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("correction", ["bonferroni", "holm", None])
def test_plot_critical_difference(correction):
    """Test plot critical difference diagram."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    fig, ax = plot_critical_difference(
        res,
        cls,
        correction=correction,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)


@pytest.mark.skipif(
    not _check_soft_dependencies("matplotlib", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_critical_difference_p_values():
    """Test plot critical difference diagram."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    cls = ["HC2", "FreshPRINCE", "InceptionT", "WEASEL-D"]
    data_full = list(univariate_equal_length)
    data_full.sort()

    res, _ = get_estimator_results_as_array(
        estimators=cls, datasets=data_full, path=data_path, include_missing=True
    )

    fig, ax, p_values = plot_critical_difference(
        res,
        cls,
        highlight=None,
        lower_better=False,
        alpha=0.05,
        width=6,
        textspace=1.5,
        reverse=True,
        return_p_values=True,
    )
    plt.gcf().canvas.draw_idle()

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
    assert isinstance(p_values, np.ndarray)
