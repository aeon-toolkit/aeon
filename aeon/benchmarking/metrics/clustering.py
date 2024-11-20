"""Clustering performance metric functions."""

__maintainer__ = ["MatthewMiddlehurst", "chrisholder"]
__all__ = ["clustering_accuracy_score"]


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def clustering_accuracy_score(y_true, y_pred):
    """Calculate clustering accuracy.

    The clustering accuracy assigns each cluster to a class by solving the linear sum
    assignment problem on the confusion matrix of the true target labels and cluster
    labels, then finds the accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target labels.
    y_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.

    Returns
    -------
    score : float
        The clustering accuracy.

    Examples
    --------
    >>> from aeon.benchmarking.metrics.clustering import clustering_accuracy_score
    >>> clustering_accuracy_score([0, 0, 1, 1], [1, 1, 0, 0]) # doctest: +SKIP
    1.0
    """
    matrix = confusion_matrix(y_true, y_pred)
    row, col = linear_sum_assignment(matrix.max() - matrix)
    s = sum([matrix[row[i], col[i]] for i in range(len(row))])
    return s / len(y_pred)
