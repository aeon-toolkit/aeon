"""Tests for the LeftSTAMPi class."""

__maintainer__ = ["ferewi"]

from unittest import mock
from unittest.mock import ANY, call

import numpy as np
import pytest

from aeon.anomaly_detection.series.distance_based._left_stampi import LeftSTAMPi
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


class FakeStumpi:
    """Fake stumpy.stumpi class."""

    def __init__(
        self,
        X,
        m,
        egress,
        normalize,
        p,
        k,
    ):
        self.X = np.asarray(X)
        self.m = m
        self.egress = egress
        self.normalize = normalize
        self.p = p
        self.k = k

        n_windows = self.X.shape[0] - self.m + 1
        self._left_P = np.arange(n_windows, dtype=np.float64)
        self._update_count = 0

    def update(self, X):
        """Fake update method."""
        self._left_P = np.append(self._left_P, self._left_P.shape[0])
        self._update_count += 1


@pytest.fixture
def mock_stumpy_pkg():
    """Mock stumpy package."""
    mock_pkg = mock.MagicMock()
    mock_pkg.stumpi_instances = []

    def fake_stumpi(X, m, egress, normalize, p, k):
        stumpi = FakeStumpi(X, m, egress, normalize, p, k)
        mock_pkg.stumpi_instances.append(stumpi)
        return stumpi

    mock_pkg.stumpi.side_effect = fake_stumpi
    with mock.patch.dict("sys.modules", {"stumpy": mock_pkg}):
        yield mock_pkg


class TestLeftSTAMPi:
    """Unit Tests for the LeftSTAMPi class."""

    @pytest.mark.skipif(
        not _check_soft_dependencies("stumpy", severity="none"),
        reason="required soft dependency stumpy not available",
    )
    def test_functional_predict_after_fit_is_independent_series(self):
        """Functional testing predict returns scores for the predict series only."""
        # given
        train = make_example_1d_numpy(n_timepoints=20, random_state=42)
        test = make_example_1d_numpy(n_timepoints=10, random_state=43)

        model = LeftSTAMPi(window_size=3, n_init_train=3)

        # when
        model = model.fit(train)
        pred = model.predict(test)

        # then
        assert pred.shape == (10,)
        assert pred.dtype == np.float64

    @pytest.mark.skipif(
        not _check_soft_dependencies("stumpy", severity="none"),
        reason="required soft dependency stumpy not available",
    )
    def test_functional_it_allows_batch_processing_via_fit_predict(self):
        """Functional testing the batch mode without mocking stumpy."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3

        model = LeftSTAMPi(window_size=4, n_init_train=5)

        # when
        pred = model.fit_predict(series)

        # then
        assert pred.shape == (20,)
        assert pred.dtype == np.float64
        assert np.argmax(pred) == 8

    def test_it_allows_batch_processing(self, mock_stumpy_pkg):
        """Unit testing the batch mode."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3

        ad = LeftSTAMPi(window_size=4, n_init_train=5)

        # when
        pred = ad.fit_predict(series)

        # then
        mock_stumpy_pkg.stumpi.assert_called_once()
        call_args, call_kwargs = mock_stumpy_pkg.stumpi.call_args
        np.testing.assert_array_equal(call_args[0], series[:5])
        assert call_kwargs == {
            "m": 4,
            "egress": False,
            "normalize": True,
            "p": 2.0,
            "k": 1,
        }
        assert mock_stumpy_pkg.stumpi_instances[-1]._update_count == 15
        assert pred.shape == (20,)
        assert pred.dtype == np.float64

    def test_predict_after_fit_returns_predict_series_length(self, mock_stumpy_pkg):
        """Unit testing predict does not append to the fit series."""
        train = make_example_1d_numpy(n_timepoints=20, random_state=42)
        test = make_example_1d_numpy(n_timepoints=10, random_state=43)
        ad = LeftSTAMPi(window_size=3, n_init_train=3)

        ad.fit(train)
        pred = ad.predict(test)

        mock_stumpy_pkg.stumpi.assert_called_once()
        call_args, call_kwargs = mock_stumpy_pkg.stumpi.call_args
        np.testing.assert_array_equal(call_args[0], test[:3])
        assert call_kwargs == {
            "m": 3,
            "egress": False,
            "normalize": True,
            "p": 2.0,
            "k": 1,
        }
        assert mock_stumpy_pkg.stumpi_instances[-1]._update_count == 7
        assert pred.shape == (10,)
        assert pred.dtype == np.float64

    def test_window_size_defaults_to_3(self, mock_stumpy_pkg):
        """Unit testing the default window size."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        # when
        ad = LeftSTAMPi()

        # then
        assert ad.window_size == 3

    def test_it_validates_window_size(self, mock_stumpy_pkg):
        """Unit testing the validation of window size."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        ad = LeftSTAMPi(window_size=2)
        # when
        with pytest.raises(
            ValueError,
            match="The window size must be at least 3 and "
            "at most the length of the time series.",
        ):
            _ = ad.fit_predict(series)

        ad = LeftSTAMPi(window_size=81)
        # when
        with pytest.raises(
            ValueError,
            match="The window size must be at least 3 and "
            "at most the length of the time series.",
        ):
            _ = ad.fit_predict(series)

    def test_it_z_normalizes_the_subsequences_by_default(self, mock_stumpy_pkg):
        """Unit testing the normalization parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        ad = LeftSTAMPi(
            window_size=5,
            n_init_train=10,
        )

        # when
        _ = ad.fit_predict(series)

        # then
        assert mock_stumpy_pkg.stumpi.call_count == 1
        mock_stumpy_pkg.stumpi.assert_has_calls(
            [
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=True,
                    p=ANY,
                    k=ANY,
                ),
            ]
        )

    def test_normalization_can_be_turned_off(self, mock_stumpy_pkg):
        """Unit testing the normalization toggle."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        ad = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False)

        # when
        _ = ad.fit_predict(series)

        # then
        assert mock_stumpy_pkg.stumpi.call_count == 1
        mock_stumpy_pkg.stumpi.assert_has_calls(
            [
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=False,
                    p=ANY,
                    k=ANY,
                )
            ]
        )

    def test_the_p_norm_can_be_changed_if_normalization_is_turned_off(
        self, mock_stumpy_pkg
    ):
        """Unit testing the pnorm parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        ad1 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False)
        ad2 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False, p=1)

        # when
        _ = ad1.fit_predict(series)
        _ = ad2.fit_predict(series)

        # then
        assert mock_stumpy_pkg.stumpi.call_count == 2
        mock_stumpy_pkg.stumpi.assert_has_calls(
            [
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=False,
                    p=2.0,
                    k=ANY,
                ),
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=False,
                    p=1.0,
                    k=ANY,
                ),
            ],
            any_order=True,
        )

    def test_the_number_of_distances_k_defaults_to_1_and_can_be_changed(
        self, mock_stumpy_pkg
    ):
        """Unit testing the k parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        ad1 = LeftSTAMPi(
            window_size=5,
            n_init_train=10,
        )
        ad2 = LeftSTAMPi(window_size=5, n_init_train=10, k=2)

        # when
        _ = ad1.fit_predict(series)
        _ = ad2.fit_predict(series)

        # then
        assert mock_stumpy_pkg.stumpi.call_count == 2
        mock_stumpy_pkg.stumpi.assert_has_calls(
            [
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=ANY,
                    p=ANY,
                    k=1,
                ),
                call(
                    ANY,
                    m=ANY,
                    egress=ANY,
                    normalize=ANY,
                    p=ANY,
                    k=2,
                ),
            ],
            any_order=True,
        )
