__author__ = ["aadya940"]
__all__ = ["SpectrogramTransformer"]

from scipy.signal import spectrogram

from aeon.transformations.series.base import BaseSeriesTransformer


class SpectrogramTransformer(BaseSeriesTransformer):
    """
    Compute a spectrogram with consecutive Fourier transforms.

    Spectrograms can be used as a way of visualizing the change
    of a nonstationary signal's frequency content over time.

    For more info checkout, checkout the scipy docs:
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html>_.

    Parameters
    ----------
    fs: float, optional
        Sampling frequency of the time series. Defaults to 1.0.
    return_onesided: boolean, optional
        If True, return a one-sided spectrum for real data.
        If False return a two-sided spectrum. Defaults to True.

    Examples
    --------
    >>> from aeon.transformations.series import SpectrogramTransformer
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.01 * fs / 2
    >>> time = np.arange(N) / float(fs)
    >>> mod = 500*np.cos(2*np.pi*0.25*time)
    >>> carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    >>> noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    >>> noise *= np.exp(-time/5)
    >>> x = carrier + noise
    >>> transformer = SpectrogramTransformer()  # doctest: +SKIP
    >>> mp = transformer.fit_transform(x)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "capability:inverse_transform": False,
    }

    def __init__(self, fs: float = 1, return_onesided=True):
        # Inputs
        self.fs = fs
        self.return_onesided = return_onesided

        # Outputs
        self.sample_frequencies = None
        self.segment_time = None
        self.spectrogram = None

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed
        y : ignored argument for interface compatibility

        Returns
        -------
        sample_frequencies: ndarray
                        Array of sample frequencies.
        segment_time:   ndarray
                    Array of segment times.
        spectrogram:     ndarray
                    Spectrogram of x. By default, the last axis of
                    spectrogram corresponds to the segment times.
        """
        self.sample_frequencies, self.segment_time, self.spectrogram = spectrogram(
            X, fs=self.fs, return_onesided=self.return_onesided
        )
        return self.sample_frequencies, self.segment_time, self.spectrogram

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        return {}
