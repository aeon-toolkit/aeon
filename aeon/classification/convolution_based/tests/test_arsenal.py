"""Arsenal test code."""

import re

import numpy as np
import pytest

from aeon.classification.convolution_based import Arsenal
from aeon.testing.data_generation import (
    make_example_2d_numpy_collection,
    make_example_3d_numpy,
)
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MultiRocket,
    Rocket,
)


def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = make_example_3d_numpy()
    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=3,
        n_kernels=20,
    )
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1


def test_arsenal():
    """Test correct rocket variant is selected."""
    X_train, y_train = make_example_2d_numpy_collection(n_cases=20, n_timepoints=50)
    afc = Arsenal(n_kernels=20, n_estimators=2)
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], Rocket)
    assert len(afc.estimators_) == 2
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocket)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocket)
    X_train, y_train = make_example_3d_numpy(n_cases=20, n_timepoints=50, n_channels=4)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocket)
    afc = Arsenal(
        n_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocket)
    afc = Arsenal(rocket_transform="fubar")
    with pytest.raises(ValueError, match="Invalid Rocket transformer: fubar"):
        afc.fit(X_train, y_train)


@pytest.mark.parametrize("verbose", [0, 1, 2])
def test_arsenal_verbosity_levels(verbose, capsys):
    """Arsenal verbosity controls summary and per-estimator progress."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        n_timepoints=24,
        n_labels=2,
        random_state=0,
    )
    arsenal = Arsenal(
        n_kernels=20,
        n_estimators=2,
        random_state=0,
        verbose=verbose,
    )

    arsenal.fit(X, y)
    output = capsys.readouterr().out

    if verbose == 0:
        assert output == ""
    else:
        assert "[Arsenal] Starting fit:" in output
        assert "[Arsenal] Finished fit: built=2" in output
        if verbose == 1:
            assert "[Arsenal] Progress: built=" in output
            assert "[Arsenal] Estimator " not in output
        else:
            assert "[Arsenal] Estimator 1/2:" in output
            assert "estimated_remaining=" in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "remaining_time_pattern"),
    [
        (1, r"contract_remaining=\d+(?:\.\d+)?s"),
        (2, r"contract_remaining=\d+m \d+s"),
        (120, r"contract_remaining=\d+h \d+m"),
    ],
)
def test_arsenal_contract_verbosity_reports_remaining_time(
    time_limit_in_minutes, remaining_time_pattern, capsys
):
    """Arsenal level-two output formats the remaining fit contract."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        n_timepoints=24,
        n_labels=2,
        random_state=0,
    )
    arsenal = Arsenal(
        n_kernels=20,
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=1,
        random_state=0,
        verbose=2,
    )

    arsenal.fit(X, y)
    output = capsys.readouterr().out

    assert "[Arsenal] Estimator 1:" in output
    assert re.search(remaining_time_pattern, output)
    assert "estimated_remaining=" not in output


@pytest.mark.parametrize(
    ("time_limit_in_minutes", "reports_progress"),
    [(0.0005, True), (120, False)],
)
def test_arsenal_contract_level_one_progress_is_rate_limited(
    time_limit_in_minutes, reports_progress, capsys
):
    """Arsenal level one limits output for short completed contracts."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        n_timepoints=24,
        n_labels=2,
        random_state=0,
    )
    arsenal = Arsenal(
        n_kernels=20,
        time_limit_in_minutes=time_limit_in_minutes,
        contract_max_n_estimators=1,
        random_state=0,
        verbose=1,
    )

    arsenal.fit(X, y)
    output = capsys.readouterr().out

    assert ("[Arsenal] Progress: built=" in output) is reports_progress
    assert "[Arsenal] Estimator " not in output


def test_parallel_verbose_fit_preserves_predictions(capsys):
    """Verbose batching preserves deterministic parallel Arsenal output."""
    X, y = make_example_3d_numpy(
        n_cases=20,
        n_timepoints=24,
        n_labels=2,
        random_state=0,
    )
    quiet = Arsenal(
        n_kernels=20,
        n_estimators=4,
        n_jobs=2,
        random_state=0,
    ).fit(X, y)
    verbose = Arsenal(
        n_kernels=20,
        n_estimators=4,
        n_jobs=2,
        random_state=0,
        verbose=2,
    ).fit(X, y)

    np.testing.assert_allclose(verbose.predict_proba(X), quiet.predict_proba(X))
    assert verbose._n_jobs == 2
    output = capsys.readouterr().out
    assert "[Arsenal] Estimator 1/4:" in output
    assert "[Arsenal] Estimator 4/4:" in output
