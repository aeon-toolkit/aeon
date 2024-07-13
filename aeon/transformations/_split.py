"""SplitsTimeSeries mixin."""

from deprecated.sphinx import deprecated


# TODO: remove in v0.12.0
@deprecated(
    version="0.11.0",
    reason="SplitsTimeSeries class will be removed in 0.12.0 or 1.0.0. Use the "
    "function split_series in utils instead.",
    category=FutureWarning,
)
class SplitsTimeSeries:
    """Split time series mixin."""

    def _split(self, X):
        """Split a time series into approximately equal intervals.

        Adopted from = https://stackoverflow.com/questions/2130016/
                       splitting-a-list-into-n-parts-of-approximately
                       -equal-length

        Parameters
        ----------
        X : a numpy array of shape = [time_n_timepoints]

        Returns
        -------
        output : a numpy array of shape = [self.n_intervals,interval_size]
        """
        avg = len(X) / float(self.n_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning) : int(beginning + avg)])
            beginning += avg

        return output
