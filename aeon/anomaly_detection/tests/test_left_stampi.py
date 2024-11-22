"""Tests for the LeftSTAMPi class."""

__maintainer__ = ["ferewi"]

from unittest import mock
from unittest.mock import ANY, call

import numpy as np
import pytest

from aeon.anomaly_detection._left_stampi import LeftSTAMPi
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
        self.X = X
        self.m = m
        self.egress = egress
        self.normalize = normalize
        self.p = p
        self.k = k

        self._left_P = np.array(
            [
                5.92,
                4.43,
                3.07,
                1.25,
                3.07,
                2.82,
                1.19,
                0.81,
                0.58,
                0.75,
                0.70,
                0.10,
                0.35,
                0.47,
                0.77,
                0.82,
                0.62,
            ]
        )

        self._update_count = 0

    def update(self, X):
        """Fake update method."""
        self._update_count += 1


@pytest.fixture
def mock_stumpy_pkg():
    """Mock stumpy package."""
    mock_pkg = mock.MagicMock()

    def fake_stumpi(X, m, egress, normalize, p, k):
        return FakeStumpi(X, m, egress, normalize, p, k)

    mock_pkg.stumpi.side_effect = fake_stumpi
    with mock.patch.dict("sys.modules", {"stumpy": mock_pkg}):
        yield mock_pkg


class TestLeftSTAMPi:
    """Unit Tests for the LeftSTAMPi class."""

    @pytest.mark.skipif(
        not _check_soft_dependencies("stumpy", severity="none"),
        reason="required soft dependency stumpy not available",
    )
    def test_functional_it_allows_batch_processing_two_step(self):
        """Functional testing the batch mode without mocking stumpy."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3

        model = LeftSTAMPi(window_size=4, n_init_train=5)

        # when
        model = model.fit(series[:5])
        pred = model.predict(series[5:])

        # then
        assert pred.shape == (20,)
        assert pred.dtype == np.float64
        assert np.argmax(pred) == 8

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
        mock_stumpy_pkg.stumpi.has_been_called_once_with(
            series[:5], m=4, egress=False, normalize=True, p=2
        )
        assert ad.mp_._update_count == 15
        assert pred.shape == (20,)
        assert pred.dtype == np.float64
        assert np.argmax(pred) == 8

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
