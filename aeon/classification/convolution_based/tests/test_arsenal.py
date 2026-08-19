"""Arsenal test code."""

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
