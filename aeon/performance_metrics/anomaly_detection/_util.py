"""Utility functions for anomaly detection performance metrics."""

__maintainer__ = ["SebastianSchmidl"]
__all__ = ["check_y"]

import warnings

import numpy as np
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d


def check_y(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    force_y_pred_continuous: bool = False,
    inf_is_1: bool = True,
    neginf_is_0: bool = True,
    nan_is_0: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the input arrays for the performance metrics.

    This function checks the input arrays for the performance metrics. You can control
    the interpretation of infinity, negative infinity, and NaN values in the anomaly
    scoring with the parameters. If the respective parameter is set to ``False``, the
    corresponding values are considered as wrong predictions (either as FP or FN
    depending on the label).

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels of shape (n_instances,).
    y_pred : np.ndarray
        Either the anomaly scores for each point of the time series or the binary
        anomaly prediction of shape (n_instances,). If ``force_y_pred_continous`` is set
        ``y_pred`` must be a float array, otherwise it must be an integer array.
    force_y_pred_continuous : bool, default false
        If ``True`` the function checks whether ``y_pred`` is a continuous float array.
        Otherwise, it is assumed to contain binary predictions as integer array. If the
        user accidentally swapped the input parameters, this functions swappes them
        back.
    inf_is_1 : bool, default True
        Whether to treat infinite values in the anomaly scores as 1.
    neginf_is_0 : bool, default True
        Whether to treat negative infinite values in the anomaly scores as 0.
    nan_is_0 : bool, default True
        Whether to treat NaN values in the anomaly scores as 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of the cleaned ``y_true`` and ``y_pred`` arrays.
    """
    y_true = np.array(y_true).copy()
    y_pred = np.array(y_pred).copy()

    # check labels
    if (
        force_y_pred_continuous
        and y_true.dtype == np.float64
        and y_pred.dtype == np.int_
    ):
        warnings.warn(
            "Assuming that y_true and y_score where permuted, because their"
            "dtypes indicate so. y_true should be an integer array and"
            "y_score a float array!",
            stacklevel=2,
        )
        return check_y(
            y_pred,
            y_true,
            force_y_pred_continuous=force_y_pred_continuous,
            inf_is_1=inf_is_1,
            neginf_is_0=neginf_is_0,
            nan_is_0=nan_is_0,
        )

    y_true = column_or_1d(y_true)
    assert_all_finite(y_true)

    # check scores
    y_pred = column_or_1d(y_pred)

    check_consistent_length([y_true, y_pred])
    if not force_y_pred_continuous and y_pred.dtype not in [np.int_, np.bool_]:
        raise ValueError(
            "When using metrics other than AUC/VUS-metrics that need binary "
            "(0 or 1) scores (like Precision, Recall or F1-Score), the scores must "
            "be integers and should only contain the values {0, 1}. Please "
            "consider applying a threshold to the scores!"
        )
    elif force_y_pred_continuous and y_pred.dtype != np.float64:
        raise ValueError(
            "When using continuous scoring metrics, the scores must be floats!"
        )

    # substitute NaNs and Infs
    nan_mask = np.isnan(y_pred)
    inf_mask = np.isinf(y_pred)
    neginf_mask = np.isneginf(y_pred)
    penalize_mask = np.full_like(y_pred, dtype=bool, fill_value=False)
    if inf_is_1:
        y_pred[inf_mask] = 1
    else:
        penalize_mask = penalize_mask | inf_mask
    if neginf_is_0:
        y_pred[neginf_mask] = 0
    else:
        penalize_mask = penalize_mask | neginf_mask
    if nan_is_0:
        y_pred[nan_mask] = 0
    else:
        penalize_mask = penalize_mask | nan_mask
    y_pred[penalize_mask] = (~np.array(y_true[penalize_mask], dtype=bool)).astype(
        np.int_
    )

    assert_all_finite(y_pred)
    return y_true, y_pred
