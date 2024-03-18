"""Unit tests for regressors deep learning random_state functionality."""

import inspect

import numpy as np
import pytest

from aeon.regression import deep_learning
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow", "tensorflow_addons"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_random_state_deep_learning_rgs():
    """Test Deep Regressor seeding."""
    random_state = 42

    X, y = make_example_3d_numpy(random_state=random_state)

    deep_rgs_classes = [
        member[1] for member in inspect.getmembers(deep_learning, inspect.isclass)
    ]

    for i in range(len(deep_rgs_classes)):
        if (
            "BaseDeepRegressor" in str(deep_rgs_classes[i])
            or "InceptionTimeRegressor" in str(deep_rgs_classes[i])
            or "LITETimeRegressor" in str(deep_rgs_classes[i])
            or "TapNetRegressor" in str(deep_rgs_classes[i])
        ):
            continue

        deep_rgs1 = deep_rgs_classes[i](random_state=random_state, n_epochs=4)
        deep_rgs1.fit(X, y)

        layers1 = deep_rgs1.training_model_.layers[1:]

        deep_rgs2 = deep_rgs_classes[i](random_state=random_state, n_epochs=4)
        deep_rgs2.fit(X, y)

        layers2 = deep_rgs2.training_model_.layers[1:]

        assert len(layers1) == len(layers2)

        for i in range(len(layers1)):
            weights1 = layers1[i].get_weights()
            weights2 = layers2[i].get_weights()

            assert len(weights1) == len(weights2)

            for j in range(len(weights1)):
                _weight1 = np.asarray(weights1[j])
                _weight2 = np.asarray(weights2[j])

                assert np.array_equal(_weight1, _weight2)
