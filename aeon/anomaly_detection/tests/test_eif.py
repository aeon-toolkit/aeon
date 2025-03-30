"""Tests for the EIF class."""

import numpy as np
import pytest

from aeon.anomaly_detection._eif import EIF
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("h2o", severity="none"),
    reason="required soft dependency h2o not available",
)
def test_eif_fit_default():
    """Test EIF _fit method with default parameters."""
    series = make_example_1d_numpy(n_timepoints=100, random_state=0)

    eif = EIF()
    eif._fit(series)

    assert hasattr(eif, "eif"), "Model was not fitted properly."
    assert hasattr(eif, "threshold_"), "Threshold was not set properly."


@pytest.mark.skipif(
    not _check_soft_dependencies("h2o", severity="none"),
    reason="required soft dependency h2o not available",
)
def test_eif_fit_empty_data():
    """Test EIF _fit method with empty input data."""
    series = np.array([])

    eif = EIF()

    with pytest.raises(ValueError, match="Training data X cannot be empty."):
        eif._fit(series)


@pytest.mark.skipif(
    not _check_soft_dependencies("h2o", severity="none"),
    reason="required soft dependency h2o not available",
)
def test_eif_fit_contamination():
    """Test EIF _fit method with different contamination levels."""
    series = make_example_1d_numpy(n_timepoints=100, random_state=0)

    eif = EIF(contamination=0.2)
    eif._fit(series)

    assert hasattr(eif, "threshold_"), "Threshold was not set properly."
    assert 0 <= eif.threshold_ <= 1, "Threshold is out of expected range."

    eif = EIF(contamination=0)
    eif._fit(series)

    assert eif.threshold_ == 0, "Threshold should be 0 when contamination is 0."


@pytest.mark.skipif(
    not _check_soft_dependencies("h2o", severity="none"),
    reason="required soft dependency h2o not available",
)
def test_eif_fit_invalid_contamination():
    """Test EIF _fit method with invalid contamination levels."""
    series = make_example_1d_numpy(n_timepoints=100, random_state=0)

    eif = EIF(contamination=-0.1)

    with pytest.raises(ValueError, match="Contamination must be between 0 and 1."):
        eif._fit(series)

    eif = EIF(contamination=1.1)

    with pytest.raises(ValueError, match="Contamination must be between 0 and 1."):
        eif._fit(series)
