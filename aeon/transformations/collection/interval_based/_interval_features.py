"""Shared helpers for extracting transformer features from interval slices.

``RandomIntervals`` and ``SupervisedIntervals`` both extract features from single
channel slices of a collection that the top-level ``fit`` has already validated.
Calling a transformer feature's public ``fit``/``fit_transform``/``transform`` on
every slice repeats that validation thousands of times per forest fit, so for aeon
collection transformers these helpers call the private methods instead, which
assume input already in the ``numpy3D`` inner type.
"""

__maintainer__ = []
__all__ = ["_fit_feature", "_fit_transform_feature", "_transform_feature"]

import numpy as np

from aeon.transformations.collection.base import BaseCollectionTransformer


def _fit_feature(feature, X, y=None, expand_fallback=True):
    """Fit a transformer feature on a single channel interval slice.

    Parameters
    ----------
    feature : BaseTransformer
        The transformer feature to fit.
    X : 2D np.ndarray of shape (n_cases, interval_length)
        A single channel interval slice, taken from input the top-level ``fit``
        has already validated.
    y : 1D np.ndarray or None, default=None
        Class labels, passed on to the feature.
    expand_fallback : bool, default=True
        Whether a feature that is not a collection transformer receives the
        slice expanded to ``numpy3D`` along with ``y``, or the 2D slice on its
        own. See the note below.

    Returns
    -------
    feature : BaseTransformer
        The fitted feature.

    Notes
    -----
    ``expand_fallback`` exists only to preserve the two callers' pre-existing
    behaviour for features that are not aeon collection transformers:
    ``RandomIntervals`` passes such a feature the expanded slice and ``y``,
    ``SupervisedIntervals`` passes the 2D slice and no ``y``. The collection
    transformer path, which is the one this module exists to speed up, is the
    same for both.
    """
    if isinstance(feature, BaseCollectionTransformer):
        return feature._fit(np.expand_dims(X, axis=1), y)
    if expand_fallback:
        return feature.fit(np.expand_dims(X, axis=1), y)
    return feature.fit(X)


def _fit_transform_feature(feature, X, y=None, expand_fallback=True):
    """Fit and transform a transformer feature on a single channel interval slice.

    Parameters
    ----------
    feature : BaseTransformer
        The transformer feature to fit and apply.
    X : 2D np.ndarray of shape (n_cases, interval_length)
        A single channel interval slice, taken from input the top-level ``fit``
        has already validated.
    y : 1D np.ndarray or None, default=None
        Class labels, passed on to the feature.
    expand_fallback : bool, default=True
        Whether a feature that is not a collection transformer receives the
        slice expanded to ``numpy3D`` along with ``y``, or the 2D slice on its
        own. See the note in ``_fit_feature``.

    Returns
    -------
    Xt : np.ndarray
        The extracted features, in whatever shape the feature returns.
    """
    if isinstance(feature, BaseCollectionTransformer):
        return feature._fit_transform(np.expand_dims(X, axis=1), y)
    if expand_fallback:
        return feature.fit_transform(np.expand_dims(X, axis=1), y)
    return feature.fit_transform(X)


def _transform_feature(feature, X, expand_fallback=True):
    """Transform a single channel interval slice with a fitted transformer feature.

    Parameters
    ----------
    feature : BaseTransformer
        The fitted transformer feature to apply.
    X : 2D np.ndarray of shape (n_cases, interval_length)
        A single channel interval slice, taken from input the top-level
        ``transform`` has already validated.
    expand_fallback : bool, default=True
        Whether a feature that is not a collection transformer receives the
        slice expanded to ``numpy3D``, or the 2D slice. See the note in
        ``_fit_feature``.

    Returns
    -------
    Xt : np.ndarray
        The extracted features, in whatever shape the feature returns.
    """
    if isinstance(feature, BaseCollectionTransformer):
        return feature._transform(np.expand_dims(X, axis=1))
    if expand_fallback:
        return feature.transform(np.expand_dims(X, axis=1))
    return feature.transform(X)
