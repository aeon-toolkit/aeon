"""Tests for shapelet plotting classes and functions."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aeon.classification.shapelet_based import (
    RDSTClassifier,
    RSASTClassifier,
    SASTClassifier,
)
from aeon.testing.data_generation import make_example_3d_numpy
from aeon.transformations.collection.shapelet_based import (
    RSAST,
    SAST,
    RandomDilatedShapeletTransform,
    RandomShapeletTransform,
)
from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import (
    ShapeletClassifierVisualizer,
    ShapeletTransformerVisualizer,
    ShapeletVisualizer,
)

CLASSIFIERS = [
    RSASTClassifier,
    SASTClassifier,
    RDSTClassifier,
]
TRANSFORMERS = [RandomShapeletTransform, RSAST, SAST, RandomDilatedShapeletTransform]
_test_shapelet_values = np.array([[1, 2, 3, 4, 3, 2, 1]])


def test_ShapeletVisualizer_init():
    """Test whether ShapeletVisualizer initialize without error."""
    shp = ShapeletVisualizer(_test_shapelet_values)
    assert_array_equal(shp.values, _test_shapelet_values)
    shp = ShapeletVisualizer(_test_shapelet_values, length=3)
    assert_array_equal(shp.values, _test_shapelet_values[:, :3])


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_ShapeletVisualizer_plot():
    """Test whether ShapeletVisualizer plot shapelets without error."""
    import matplotlib.pyplot as plt

    shp = ShapeletVisualizer(_test_shapelet_values)
    fig = shp.plot()
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_ShapeletVisualizer_plot_on_X():
    """Test whether ShapeletVisualizer plot shapelets on X without error."""
    import matplotlib.pyplot as plt

    shp = ShapeletVisualizer(_test_shapelet_values)
    X = make_example_3d_numpy(n_cases=1)[0][0]
    fig = shp.plot_on_X(X)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_ShapeletVisualizer_plot_distance_vector():
    """Test whether ShapeletVisualizer plot distance vectors runs without error."""
    import matplotlib.pyplot as plt

    shp = ShapeletVisualizer(_test_shapelet_values)
    X = make_example_3d_numpy(n_cases=1)[0][0]
    fig = shp.plot_distance_vector(X)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("transformer_class", TRANSFORMERS)
def test_ShapeletTransformerVisualizer(transformer_class):
    """Test whether ShapeletTransformerVisualizer runs without error."""
    import matplotlib.pyplot as plt

    X, y = make_example_3d_numpy()
    shp_transformer = transformer_class(**transformer_class._get_test_params()).fit(
        X, y
    )
    shp_vis = ShapeletTransformerVisualizer(shp_transformer)

    fig = shp_vis.plot(0)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()

    fig = shp_vis.plot_on_X(0, X[0])
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()

    fig = shp_vis.plot_distance_vector(0, X[0])
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib"], severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("classifier_class", CLASSIFIERS)
def test_ShapeletClassifierVisualizer(classifier_class):
    """Test whether ShapeletClassifierVisualizer runs without error."""
    import matplotlib.pyplot as plt

    X, y = make_example_3d_numpy()
    shp_transformer = classifier_class(**classifier_class._get_test_params()).fit(X, y)
    shp_vis = ShapeletClassifierVisualizer(shp_transformer)

    fig = shp_vis.plot(0)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()

    fig = shp_vis.plot_on_X(0, X[0])
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()

    fig = shp_vis.plot_distance_vector(0, X[0])
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig, plt.Figure)
    plt.close()

    fig = shp_vis.visualize_shapelets_one_class(X, y, 0)
    plt.gcf().canvas.draw_idle()
    assert isinstance(fig[0], plt.Figure)
    plt.close()
