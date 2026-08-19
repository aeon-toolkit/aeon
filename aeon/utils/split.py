"""Split function."""


def split_series(X, n_intervals):
    """Split a time series into approximately equal intervals.

    Adopted from = https://stackoverflow.com/questions/2130016/
                   splitting-a-list-into-n-parts-of-approximately
                   -equal-length

    Parameters
    ----------
    X : a numpy array of shape = [n_timepoints]

    Returns
    -------
    output : a numpy array of shape = [self.n_intervals,interval_size]
    """
    avg = len(X) / float(n_intervals)
    output = []
    beginning = 0.0

    while beginning < len(X):
        output.append(X[int(beginning) : int(beginning + avg)])
        beginning += avg

    return output
