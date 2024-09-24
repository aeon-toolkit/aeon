"""Tests for the LaftSTAMPi class."""

__maintainer__ = ["FerdinandRewicki"]
import numpy as np
import pytest

from aeon.anomaly_detection._left_stampi import LeftSTAMPi
from aeon.testing.data_generation import make_example_1d_numpy
from aeon.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("stumpy", severity="none"),
    reason="required soft dependency stumpy not available",
)
class TestLeftSTAMPi:
    """Unit Tests for the LeftSTAMPi class."""

    def test_functional_it_allows_batch_processing(self):
        """Functional testing the batch mode without mocking stumpy."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3

        model = LeftSTAMPi(window_size=4, n_init_train=5)

        # when
        pred = model.fit_predict(series)

        # then
        assert pred.shape == (20,)
        assert pred.dtype == np.float_
        assert np.argmax(pred) == 8

    def test_functional_it_allows_stream_processing_by_incremental_updates(self):
        """Functional testing the steam mode without mocking stumpy."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3
        init_series = series[:5]
        streaming_data = series[5:]

        model = LeftSTAMPi(
            window_size=4,
        )

        # when
        model = model.fit(init_series)
        pred = np.array([])
        for x in streaming_data:
            pred = model.predict(np.array([x]))

        # then
        assert pred.shape == (20,)
        assert pred.dtype == np.float_
        assert np.argmax(pred) == 8

    def test_it_allows_batch_processing(self, mocker):
        """Unit testing the batch mode."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3

        stumpi_mock = mocker.patch("stumpy.stumpi")
        stumpi_instance = mocker.Mock()
        stumpi_mock.return_value = stumpi_instance
        stumpi_instance._left_P = np.array(
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

        ad = LeftSTAMPi(window_size=4, n_init_train=5)

        # when
        pred = ad.fit_predict(series)

        # then
        stumpi_mock.has_been_called_once_with(
            series[:5], m=4, egress=False, normalize=True, p=2
        )
        assert stumpi_instance.update.call_count == 15
        assert pred.shape == (20,)
        assert pred.dtype == np.float_
        assert np.argmax(pred) == 8

    def test_it_allows_stream_processing_by_incremental_updates(self, mocker):
        """Unit testing the stream mode."""
        # given
        series = make_example_1d_numpy(n_timepoints=20, random_state=42)
        series[7:10] += 3
        init_series = series[:5]
        streaming_data = series[5:]

        stumpi_mock = mocker.patch("stumpy.stumpi")
        stumpi_instance = mocker.Mock()
        stumpi_mock.return_value = stumpi_instance
        stumpi_instance._left_P = np.array(
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

        ad = LeftSTAMPi(
            window_size=4,
        )

        # when
        ad = ad.fit(init_series)
        for x in streaming_data:
            pred = ad.predict(np.array([x]))

        # then
        stumpi_mock.has_been_called_once_with(
            series[:5], m=4, egress=False, normalize=True, p=2
        )
        assert stumpi_instance.update.call_count == 15
        assert pred.shape == (20,)
        assert pred.dtype == np.float_
        assert np.argmax(pred) == 8

    def test_window_size_defaults_to_3(self):
        """Unit testing the default window size."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2

        # when
        ad = LeftSTAMPi()

        # then
        assert ad.window_size == 3

    def test_it_validates_window_size(self):
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
            _ = ad.fit(series)

        ad = LeftSTAMPi(window_size=81)
        # when
        with pytest.raises(
            ValueError,
            match="The window size must be at least 3 and "
            "at most the length of the time series.",
        ):
            _ = ad.fit_predict(series)

    def test_it_z_normalizes_the_subsequences_by_default(self, mocker):
        """Unit testing the normalization parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2
        stumpi_stub = mocker.patch("stumpy.stumpi")

        ad1 = LeftSTAMPi(
            window_size=5,
        )
        ad2 = LeftSTAMPi(
            window_size=5,
            n_init_train=10,
        )

        # when
        _ = ad1.fit(series[:10])
        _ = ad2.fit_predict(series)

        # then
        assert stumpi_stub.call_count == 2
        stumpi_stub.assert_has_calls(
            [
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=True,
                    p=mocker.ANY,
                    k=mocker.ANY,
                ),
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=True,
                    p=mocker.ANY,
                    k=mocker.ANY,
                ),
            ]
        )

    def test_normalization_can_be_turned_off(self, mocker):
        """Unit testing the normalization toggle."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2
        stumpi_stub = mocker.patch("stumpy.stumpi")

        ad1 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False)
        ad2 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False)

        # when
        _ = ad1.fit(series[:10])
        _ = ad2.fit_predict(series)

        # then
        assert stumpi_stub.call_count == 2
        stumpi_stub.assert_has_calls(
            [
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=False,
                    p=mocker.ANY,
                    k=mocker.ANY,
                ),
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=False,
                    p=mocker.ANY,
                    k=mocker.ANY,
                ),
            ]
        )

    def test_the_p_norm_can_be_changed_if_normalization_is_turned_off(self, mocker):
        """Unit testing the pnorm parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2
        stumpi_stub = mocker.patch("stumpy.stumpi")

        ad1 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False)
        ad2 = LeftSTAMPi(window_size=5, n_init_train=10, normalize=False, p=1)

        # when
        _ = ad1.fit(series[:10])
        _ = ad2.fit_predict(series)

        # then
        assert stumpi_stub.call_count == 2
        stumpi_stub.assert_has_calls(
            [
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=mocker.ANY,
                    p=2,
                    k=mocker.ANY,
                ),
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=mocker.ANY,
                    p=1,
                    k=mocker.ANY,
                ),
            ]
        )

    def test_the_number_of_distances_k_defaults_to_1_and_can_be_changed(self, mocker):
        """Unit testing the k parameter."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2
        stumpi_stub = mocker.patch("stumpy.stumpi")

        ad1 = LeftSTAMPi(
            window_size=5,
            n_init_train=10,
        )
        ad2 = LeftSTAMPi(window_size=5, n_init_train=10, k=2)

        # when
        _ = ad1.fit(series[:10])
        _ = ad2.fit_predict(series)

        # then
        assert stumpi_stub.call_count == 2
        stumpi_stub.assert_has_calls(
            [
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=mocker.ANY,
                    p=mocker.ANY,
                    k=1,
                ),
                mocker.call(
                    mocker.ANY,
                    m=mocker.ANY,
                    egress=mocker.ANY,
                    normalize=mocker.ANY,
                    p=mocker.ANY,
                    k=2,
                ),
            ]
        )

    def test_it_checks_soft_dependencies(self, mocker):
        """Unit testing the dependency check."""
        # given
        series = make_example_1d_numpy(n_timepoints=80, random_state=42)
        series[50:58] += 2
        ad = LeftSTAMPi(window_size=5, n_init_train=None)
        deps_checker_stub = mocker.patch(
            "aeon.utils.validation._dependencies._check_soft_dependencies"
        )

        # when
        ad.fit(series[:10])

        # then
        deps_checker_stub.assert_called_once_with("stumpy", severity="error", obj=ad)
