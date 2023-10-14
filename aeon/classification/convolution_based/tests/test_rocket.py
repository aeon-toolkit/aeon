"""Rocket test code."""
import pytest

from aeon.classification.convolution_based import RocketClassifier
from aeon.transformations.collection.convolution_based import (
    MiniRocket,
    MiniRocketMultivariate,
    MultiRocket,
    MultiRocketMultivariate,
    Rocket,
)
from aeon.utils._testing.collection import make_2d_test_data, make_3d_test_data


def test_rocket():
    """Test correct rocket variant is selected."""
    X_train, y_train = make_2d_test_data(n_cases=20, n_timepoints=50)
    rocket = RocketClassifier(num_kernels=20)
    rocket.fit(X_train, y_train)
    assert isinstance(rocket._transformer, Rocket)
    rocket = RocketClassifier(
        num_kernels=100, rocket_transform="minirocket", max_dilations_per_kernel=2
    )
    rocket.fit(X_train, y_train)
    assert isinstance(rocket._transformer, MiniRocket)
    rocket = RocketClassifier(
        num_kernels=100, rocket_transform="multirocket", max_dilations_per_kernel=2
    )
    rocket.fit(X_train, y_train)
    assert isinstance(rocket._transformer, MultiRocket)
    X_train, y_train = make_3d_test_data(n_cases=20, n_timepoints=50, n_channels=4)
    rocket = RocketClassifier(
        num_kernels=100, rocket_transform="minirocket", max_dilations_per_kernel=2
    )
    rocket.fit(X_train, y_train)
    assert isinstance(rocket._transformer, MiniRocketMultivariate)
    rocket = RocketClassifier(
        num_kernels=100, rocket_transform="multirocket", max_dilations_per_kernel=2
    )
    rocket.fit(X_train, y_train)
    assert isinstance(rocket._transformer, MultiRocketMultivariate)
    rocket = RocketClassifier(
        num_kernels=100, rocket_transform="fobar", max_dilations_per_kernel=2
    )
    with pytest.raises(ValueError, match="Invalid Rocket transformer"):
        rocket.fit(X_train, y_train)
