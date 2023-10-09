"""Arsenal test code."""
import pytest

from aeon.classification.convolution_based import Arsenal
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)
from aeon.utils._testing.collection import make_2d_test_data, make_3d_test_data


def test_contracted_arsenal():
    """Test of contracted Arsenal on unit test data."""
    # load unit test data
    X_train, y_train = make_2d_test_data()
    # train contracted Arsenal
    arsenal = Arsenal(
        time_limit_in_minutes=0.25,
        contract_max_n_estimators=5,
        num_kernels=20,
        random_state=0,
    )
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1
    X_train, y_train = make_3d_test_data()
    arsenal.fit(X_train, y_train)
    assert len(arsenal.estimators_) > 1


def test_arsenal():
    """Test correct rocket variant is selected."""
    X_train, y_train = make_2d_test_data(n_cases=20, n_timepoints=50)
    afc = Arsenal(num_kernels=20, n_estimators=2)
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], Rocket)
    assert len(afc.estimators_) == 2
    afc = Arsenal(
        num_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocket)
    afc = Arsenal(
        num_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocket)
    X_train, y_train = make_3d_test_data(n_cases=20, n_timepoints=50, n_channels=4)
    afc = Arsenal(
        num_kernels=100,
        rocket_transform="minirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MiniRocketMultivariate)
    afc = Arsenal(
        num_kernels=100,
        rocket_transform="multirocket",
        max_dilations_per_kernel=2,
        n_estimators=2,
    )
    afc.fit(X_train, y_train)
    for i in range(afc.n_estimators):
        assert isinstance(afc.estimators_[i].steps[0][1], MultiRocketMultivariate)
    afc = Arsenal(rocket_transform="fubar")
    with pytest.raises(ValueError, match="Invalid Rocket transformer: fubar"):
        afc.fit(X_train, y_train)
