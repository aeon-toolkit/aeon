import numpy as np
from sklearn.linear_model import RidgeClassifierCV  # Normalize parameter is need set to True.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs

class TSFreshFeatureExtractor(BaseCollectionTransformer):
    pass


def series_set_dilation(seriesX, d_rate=1):
    """
    Each series of the time series set is mapped by dilation mapping with 
    the same dilation rate to output the dilated time series set.

    Parameters
    ----------
    seriesX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints] 
        The set of three dimensional time series set to be dilated.
    d_rate : int, default=1
        dilation rate.
    
    References
    ----------
    .. [1] P. Schaefer and U. Leser, “WEASEL 2.0: a random dilated dictionary 
    transform for fast, accurate and memory constrained time series classification” 
    Machine Learning, vol. 112, no. 12, pp. 4763–4788, Dec.(2024).
    """
    
    n_cases, n_channels, _ = seriesX.shape[:]
    seriesXE = np.zeros_like(seriesX)  # Initializing the dilated time series set
    
    for i in range(n_cases):
        for j in range(n_channels):
            series_ = []
            for d in range(d_rate):
                series_.append(seriesX[i, j, d::d_rate])
            seriesXE[i, j, :] = np.hstack(series_)
            
    return seriesXE  # Return the dilated time series set


def fhan(x1, x2, r, h0):
    """
    The fhan function for calculating differential signal based on optimal 
    control in tracking differentiator.

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
    
    References
    ----------
    .. [1] J. Han, “From PID to active disturbance rejection control” IEEE Trans.
    Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
    """
    
    d = r * h0
    d0 = d * h0
    y = x1 + h0 * x2    # Computing the differential signal
    a0 = np.sqrt(d*d + 8*r*np.abs(y))

    if np.abs(y) > d0:
        a = x2 + (a0-d) / 2.0 * np.sign(y)
    else:
        a = x2 + y/h0

    if np.abs(a) <= d:  # Computing the input u of observer
        u = -r * a / d
    else:
        u = -r * np.sign(a)

    return u, y # Return input u of observer, and differential signal y


def td(signal, r=100, k=3, h=1):
    """
    The tracking differentiator with a adjustable filter factor to 
    compute a differential signal

    Parameters
    ----------
    signal : 1D np.ndarray of shape = [n_timepoints] 
        Original time series
    r : float
        Velocity factor used to control tracking speed.
    k: float
        filter factor.
    h : float
        Step size.
    
    References
    ----------
    .. [1] J. Han, “From PID to active disturbance rejection control” IEEE Trans.
    Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
    """
    x1 = signal[0]   # Initializing state 1
    x2 = -(signal[1] - signal[0]) / h   # Initializing state 2

    h0 = k * h
    signalTD = np.zeros(len(signal))
    dSignal = np.zeros(len(signal))
    for i in range(len(signal)):
        v = signal[i]
        x1k = x1    
        x2k = x2    
        x1 = x1k + h*x2k  # Update state 1
        u, y = fhan(x1k-v, x2k, r, h0)   # Update input u of observer and differential signal y
        x2 = x2k + h * u  # Update state 2
        dSignal[i] = y 
        signalTD[i] = x1 
    dSignal = -dSignal / h0  # Scale transform

    return dSignal[1:]  # Return the differential signal

def series_transform(seriesX, mode=1, k1=2, k2=2):  # 时间序列变换，包括
    """
    Each series of the time series set is transformed by tracking differentiator with 
    a adjustable filter factor to output the differential time series set.

    Parameters
    ----------
    seriesX : 3D np.ndarray of shape = [n_cases, n_channels, n_timepoints] 
        The set of three dimensional time series set to be dilated.
    mode : int, default=1
        The flag bit of a first-order or second-order derivative is used.
        Computing the first-order derivative when mode=1, 
        and computing the second-order derivative when mode=2
    k1 : float, default=2
        filter factor 1 of the tracking differentiator 1.
    k2 : float, default=2
        filter factor 2 of the tracking differentiator 2.
        This parameter is invalid when mode=2.
    
    References
    ----------
    .. [1] J. Han, “From PID to active disturbance rejection control” IEEE Trans.
    Ind. Electron., vol. 56, no. 3, pp. 900-906, Mar. (2009)..
    """

    n_cases, n_channels, n_timepoints = seriesX.shape[:]

    if mode == 1:  # First-order derivative
        seriesFX = np.zeros((n_cases, n_channels, n_timepoints-1))
        for i in range(n_cases):
            for j in range(n_channels):
                seriesFX[i, j, :] = td(seriesX[i, j, :], k=k1)
                seriesFX[i, j, :] = scale(seriesFX[i, j, :])
        return seriesFX  # Return the first-order differential time series set
    
    if mode == 2:  # Second-order derivative
        seriesSX = np.zeros((n_cases, n_channels, n_timepoints-2))
        for i in range(n_cases):
            for j in range(n_channels):
                seriesF_ = td(seriesX[i, j, :], k=k1)
                seriesSX[i, j, :] = td(seriesF_, k=k2)
                seriesSX[i, j, :] = scale(seriesSX[i, j, :])
        return seriesSX  # Return the second-order differential time series set
    
def hard_voting(testYList):
    """
    The predicted labels are obtained by hard voting to process the labels matrix
    from multiple classifiers.

    Parameters
    ----------
    testYList : 2D np.ndarray of shape = [n_classifierss, n_cases]
    """
    uniqueY = np.unique(testYList)  # Holds the label for each class
    n_classes = len(uniqueY)  # Number of classes
    n_classifiers, n_cases= testYList.shape[:]  # Number of classifiers, Number of cases
    testVY = np.zeros(n_cases, int)  # 1 * n_cases, Initializing the predicted labels
   
    testWeightArray = np.zeros((n_classes, n_cases))  # n_classes * n_cases, Label weight matrix for samples
    for i in range(n_cases):
        for j in range(n_classifiers):
            label_ = testYList[j, i]
            index_ = np.arange(n_classes)[uniqueY==label_]
            testWeightArray[index_, i] += 1  # The label weight for the sample is + 1 
    for i in range(n_cases):  # Predicting each sample label
        testVY[i] = uniqueY[np.argmax(testWeightArray[:, i])]  # The label is predicted to be the most weighted
    return testVY  # return the predicted labels