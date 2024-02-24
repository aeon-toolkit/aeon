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

    kwargs: Keyword arguments passed to `scipy.signal.spectogram`
            Checkout <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html>_.


    Examples
    --------
    >>> from aeon.transformations.series import SpectogramTransformer
    >>> import numpy as np
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
    >>> transformer = SpectrogramTransformer(fs)  # doctest: +SKIP
    >>> mp = transformer.fit_transform(x)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
    }

    def __init__(self, fs: float = 1):
        # Inputs
        self.fs = fs

        # Outputs
        self.sample_frequencies = None
        self.segment_time = None
        self.spectogram = None

        super().__init__()

    def _transform(self, X, y=None, **kwargs):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : np.ndarray
            1D time series to be transformed
        y : ignored argument for interface compatibility
        kwargs : kwargs to be passed to `scipy.signal.spectogram`

        Returns
        -------
        sample_frequencies: ndarray
                        Array of sample frequencies.
        segment_time:   ndarray
                    Array of segment times.
        spectogram:     ndarray
                    Spectrogram of x. By default, the last axis of
                    spectogram corresponds to the segment times.
        """

        self.sample_frequencies, self.segment_time, self.spectogram = spectrogram(
            X, fs=self.fs, **kwargs
        )
        return self.sample_frequencies, self.segment_time, self.spectogram

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
