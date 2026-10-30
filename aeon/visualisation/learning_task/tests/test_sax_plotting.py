"""Test the plotting functions for sax."""

from types import SimpleNamespace

import numpy as np
import pytest

from aeon.utils.validation._dependencies import _check_soft_dependencies
from aeon.visualisation import plot_sax_representation


@pytest.mark.skipif(
    not _check_soft_dependencies(["matplotlib", "seaborn"], severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_plot_sax_representation_smoke():
    """Test the plot_sax_representation function with a simple example."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    X = np.random.randn(1, 1, 20)
    X_inverse = X.copy()
    X_sax = np.array([[["a", "b", "c", "d"]]], dtype=object)

    sax = SimpleNamespace(
        n_segments=4,
        breakpoints=np.array([-0.67, 0.0, 0.67]),
        distribution_params_={"scale": 1.0},
        window_size=20,
        stride=None,
    )

    fig, ax = plot_sax_representation(
        X=X,
        X_sax=X_sax,
        X_inverse=X_inverse,
        sax=sax,
    )

    assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)

    plt.close()
