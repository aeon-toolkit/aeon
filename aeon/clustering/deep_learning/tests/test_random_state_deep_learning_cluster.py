"""Unit tests for clusterer deep learning random_state functionality."""

import inspect

import numpy as np
import pytest

from aeon.clustering import deep_learning
from aeon.testing.utils.data_gen import make_example_3d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies

__maintainer__ = ["hadifawaz1999"]


@pytest.mark.skipif(
    not _check_soft_dependencies(["tensorflow", "tensorflow_addons"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_random_state_deep_learning_clr():
    """Test Deep Clusterer seeding."""
    random_state = 42

    X, _ = make_example_3d_numpy(random_state=random_state)

    deep_clr_classes = [
        member[1] for member in inspect.getmembers(deep_learning, inspect.isclass)
    ]

    for i in range(len(deep_clr_classes)):
        if "BaseDeepClusterer" in str(deep_clr_classes[i]):
            continue

        deep_clr1 = deep_clr_classes[i](
            n_clusters=2, random_state=random_state, n_epochs=4
        )
        deep_clr1.fit(X)

        layers1 = deep_clr1.training_model_.layers[1:]

        deep_clr2 = deep_clr_classes[i](
            n_clusters=2, random_state=random_state, n_epochs=4
        )
        deep_clr2.fit(X)

        layers2 = deep_clr2.training_model_.layers[1:]

        assert len(layers1) == len(layers2)

        for i in range(len(layers1)):
            weights1 = layers1[i].get_weights()
            weights2 = layers2[i].get_weights()

            assert len(weights1) == len(weights2)

            for j in range(len(weights1)):
                _weight1 = np.asarray(weights1[j])
                _weight2 = np.asarray(weights2[j])

                assert np.array_equal(_weight1, _weight2)
