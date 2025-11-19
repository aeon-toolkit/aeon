"""
Core utility functions for the Multiview Enhanced Characteristics (Mecha) Classifier.

This module houses the necessary logic for the Tracking Differentiator (TD)
series transformation and the bidirectional dilation/interleaving shuffling
mechanisms required for Mecha's diverse feature extraction.
"""

__maintainer__ = []
__all__ = [
    "series_transform",
    "dilated_fres_extract",
    "interleaved_fres_extract",
    "hard_voting",
]

import warnings

import numpy as np
from sklearn.preprocessing import scale

from aeon.transformations.collection.feature_based._catch22 import Catch22
from aeon.transformations.collection.feature_based._tsfresh import (
    TSFresh,
    TSFreshRelevant,
)

warnings.filterwarnings("ignore")


def fhan(x1: float, x2: float, r: float, h0: float) -> tuple[float, float]:
    """
    Compute a differential signal using the tracking differentiator.

    Parameters
    ----------
    x1 : float
        State 1 of the observer.
    x2 : float
        State 2 of the observer.
    r: float
        Velocity factor used to control tracking speed.
    h0 : float
        Step size.
    """
    d = r * h0
    d0 = d * h0
    y = x1 + h0 * x2  # Computing the differential signal
    a0 = np.sqrt(d * d + 8 * r * np.abs(y))

    if np.abs(y) > d0:
        a = x2 + (a0 - d) / 2.0 * np.sign(y)
    else:
        a = x2 + y / h0

    if np.abs(a) <= d:  # Computing the input u of observer
        u = -r * a / d
    else:
        u = -r * np.sign(a)

    return u, y  # Return input u of observer, and differential signal y


def td(signal: np.ndarray, r: float = 100, k: float = 3, h: float = 1) -> np.ndarray:
    """
    Compute a differential signal using the tracking differentiator.

    Parameters
    ----------
    signal : 1D np.ndarray of shape = [n_timepoints]
        Original time series.
    r : float, default=100
        Velocity factor used to control tracking speed.
    k: float, default=3
        Filter factor.
    h : float, default=1
        Step size.

    Returns
    -------
    dSignal : 1D np.ndarray of shape = [n_timepoints-1]
        The first-order differential signal.
    """
    x1 = signal[0]  # Initializing state 1
    x2 = -(signal[1] - signal[0]) / h  # Initializing state 2

    h0 = k * h
    signalTD = np.zeros(len(signal))
    dSignal = np.zeros(len(signal))
    for i in range(len(signal)):
        v = signal[i]
        x1k = x1
        x2k = x2
        x1 = x1k + h * x2k  # Update state 1
        u, y = fhan(
            x1k - v, x2k, r, h0
        )  # Update input u of observer and differential signal y
        x2 = x2k + h * u  # Update state 2
        dSignal[i] = y
        signalTD[i] = x1
    dSignal = -dSignal / h0  # Scale transform

    return dSignal[1:]  # Return the differential signal


def series_transform(seriesX: np.ndarray, k1: float) -> np.ndarray:
    """
    Transform each series using the Tracking Differentiator.

    Parameters
    ----------
    seriesX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints]
        The set of time series to be transformed.
    k1: float
        Filter factor of the TD.

    Returns
    -------
    seriesFX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints-1]
        The first-order differential time series set.
    """
    n_cases, n_channels, n_timepoints = seriesX.shape[:]

    seriesFX = np.zeros((n_cases, n_channels, n_timepoints - 1))
    for i in range(n_cases):
        for j in range(n_channels):
            seriesFX[i, j, :] = td(seriesX[i, j, :], k=k1)
            seriesFX[i, j, :] = scale(seriesFX[i, j, :])
    return seriesFX  # Return the first-order differential time series set


def bidirect_dilation_mapping(seriesX: np.ndarray, max_rate: int = 16) -> np.ndarray:
    """
    Obtain a list of series indices.

    Indice are bidirectionally dilated under the exponential shuffling rates.
    """
    _, _, n_timepoints = (
        seriesX.shape[0],
        seriesX.shape[1],
        seriesX.shape[2],
    )
    max_power = np.min([int(np.log2(max_rate)), int(np.log2(n_timepoints - 1)) - 3])
    max_power = np.max([1, max_power])  # Guaranteed to be positive
    dilation_rates = 2 ** np.arange(1, max_power + 1)  # Shuffling rates

    indexList0 = np.arange(n_timepoints)  # The index of the raw series
    indexListF = []  # Initialize the index list for forward dilation
    indexListB = []  # Initialize the index list for backward dilation
    for i in range(len(dilation_rates)):  # Perform dilation at each shuffling rate
        rate_ = dilation_rates[i]  # shuffling rate
        # (1) Forward dilation mapping
        index_ = np.array([])
        for j in range(rate_):  # Rearrange index
            index_ = np.concatenate((index_, indexList0[j::rate_])).astype(int)
        indexListF.append(index_)
        # (2) Backward dilation mapping
        index_ = np.array([])
        for j in range(rate_):  # Rearrange index
            index_ = np.concatenate((indexList0[j::rate_], index_)).astype(int)
        indexListB.append(index_)

    indexList = np.vstack((indexListF, indexListB))

    return indexList


def bidirect_interleaving_mapping(
    seriesX: np.ndarray, max_rate: int = 16
) -> np.ndarray:
    """
    Obtain a list of series indices.

    Indice are bidirectionally interleaved under the exponential shuffling rates.
    """
    _, _, n_timepoints = (
        seriesX.shape[0],
        seriesX.shape[1],
        seriesX.shape[2],
    )
    max_power = np.min([int(np.log2(max_rate)), int(np.log2(n_timepoints - 1)) - 3])
    max_power = np.max([1, max_power])  # Guaranteed to be positive
    dilation_rates = 2 ** np.arange(1, max_power + 1)  # Shuffling rates

    indexList0 = np.arange(n_timepoints)  # The index of the raw series
    indexListF = []  # Initialize the index list for forward interleaving
    indexListB = []  # Initialize the index list for backward interleaving
    for i in range(len(dilation_rates)):  # Perform interleaving at each shuffling rate
        rate_ = dilation_rates[i]  # shuffling rate
        index_ = np.array([])
        segmentNs = np.zeros(rate_, int)
        segmentIndex = []
        start = 0
        for j in range(rate_):  # Get the length and index of each segment
            segmentNs[j] = len(indexList0[j::rate_])
            segmentIndex.append(np.arange(start, start + segmentNs[j]))
            start += segmentNs[j]
        # (1) Forward interleaving mapping
        index_ = np.array([])
        for j in range(len(segmentIndex[0])):  # Rearrange index
            for k in range(rate_):  # Take a point from each segment
                index_ = np.concatenate((index_, [segmentIndex[k][j]])).astype(int)
                if len(index_) == len(indexList0):
                    break
            if len(index_) == len(indexList0):
                break
        indexListF.append(index_)

        # (2) Backward interleaving mapping
        index_ = np.array([])
        segmentNs = segmentNs[::-1]
        segmentIndex = []
        start = 0
        for j in range(rate_):  # Get the length and index of each segment
            segmentIndex.append(np.arange(start, start + segmentNs[j]))
            start += segmentNs[j]

        for j in range(len(segmentIndex[-1])):  # Rearrange index
            for k in range(rate_):  # Take a point from each segment
                if np.abs(j) >= len(segmentIndex[-k - 1]):
                    continue
                index_ = np.concatenate((index_, [segmentIndex[-k - 1][j]])).astype(int)
        indexListB.append(index_)

    indexList = np.vstack((indexListF, indexListB))

    return indexList


def _fres_extract(
    seriesX: np.ndarray, all_indices: np.ndarray, basic_extractor: str
) -> np.ndarray:
    """Handle Catch22/TSFresh feature extraction on re-indexed series views."""
    if basic_extractor == "Catch22":
        extractor = Catch22(catch24=False, replace_nans=True)
    elif basic_extractor == "TSFresh":
        extractor = TSFresh(default_fc_parameters="efficient")
    elif basic_extractor == "TSFreshRelevant":
        extractor = TSFreshRelevant(default_fc_parameters="efficient")
    else:
        raise ValueError(
            f"basic_extractor must be one of 'Catch22',"
            f"'TSFresh', or 'TSFreshRelevant'. Found: {basic_extractor}"
        )

    featureXList = []

    for index_ in all_indices:
        # Re-order the time points based on the shuffled index
        X_shuffled = seriesX[:, :, index_]

        # Fit and transform using the chosen base feature extractor
        extractor.fit(X_shuffled)
        featureX_ = np.asarray(extractor.transform(X_shuffled))
        featureXList.append(featureX_)

    featureX = np.hstack(featureXList)
    return featureX


def dilated_fres_extract(
    seriesX: np.ndarray, max_rate: int = 16, basic_extractor: str = "TSFresh"
) -> np.ndarray:
    """Extract statistical feature vectors based on dilation mapping."""
    indexList = bidirect_dilation_mapping(seriesX, max_rate=max_rate)

    # Add the index of the raw series (0 to n_timepoints-1) as the first view
    n_timepoints = seriesX.shape[2]
    full_index = np.arange(n_timepoints)
    all_indices = np.vstack((full_index[np.newaxis, :], indexList))

    return _fres_extract(seriesX, all_indices, basic_extractor)


def interleaved_fres_extract(
    seriesX: np.ndarray, max_rate: int = 16, basic_extractor: str = "TSFresh"
) -> np.ndarray:
    """Extract features based on interleaved mapping."""
    indexList = bidirect_interleaving_mapping(seriesX, max_rate=max_rate)
    return _fres_extract(seriesX, indexList, basic_extractor)


def hard_voting(testYList: np.ndarray) -> np.ndarray:
    """Obtain predicted labels by hard voting from multiple classifiers."""
    uniqueY = np.unique(testYList)
    n_classes = len(uniqueY)
    n_classifiers, n_cases = testYList.shape[:]
    testVY = np.zeros(n_cases, int)

    testWeightArray = np.zeros((n_classes, n_cases))
    for i in range(n_cases):
        for j in range(n_classifiers):
            label_ = testYList[j, i]
            index_ = np.where(uniqueY == label_)[0]
            if len(index_) > 0:
                testWeightArray[index_[0], i] += 1

    for i in range(n_cases):
        testVY[i] = uniqueY[np.argmax(testWeightArray[:, i])]
    return testVY
