"""WEASEL test code."""

import numpy as np
from scipy.stats import zscore
from sklearn.decomposition import PCA

from aeon.datasets import load_italy_power_demand
from aeon.transformations.collection.dictionary_based._sfa_fast import (
    _fast_fourier_transform,
)


# Function to compute explained variance on any dataset
def explain_variance(X, model, n_components):
    result = np.zeros(n_components)
    for ii in range(n_components):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)
        result[ii] = (1 - (np.linalg.norm(X_approx_ii - X) / np.linalg.norm(
            X - model.mean_)) ** 2
        )
    return result


def test_explained_variance():
    n_components = 8

    X_train, y_train = load_italy_power_demand(split="train", return_type="numpy2d")
    X_test, y_test = load_italy_power_demand(split="test", return_type="numpy2d")
    X_train = zscore(X_train, axis=1)
    X_test = zscore(X_test, axis=1)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # DFTs
    n = X_train.shape[-1]
    dft_length = n // 2 - n % 2
    inverse_sqrt_win_size = 1.0 / np.sqrt(n)
    norm = True
    norm_std = True

    X_train_dfts = _fast_fourier_transform(
        X_train, norm, dft_length, inverse_sqrt_win_size, norm_std=norm_std
    )

    X_test_dfts = _fast_fourier_transform(
        X_test, norm, dft_length, inverse_sqrt_win_size, norm_std=norm_std
    )

    # PCA
    pca_transform = PCA(n_components=n_components, svd_solver="auto").fit(X_train_dfts)
    X_dft = pca_transform.transform(X_train_dfts)
    explained_variance = pca_transform.explained_variance_ratio_

    test_explained_var = explain_variance(X_test_dfts, pca_transform, X_dft.shape[-1])

    print("--Results--")
    print(
        f"DFT Compression:    {X_dft.shape[-1]} of {X_train.shape[-1]}\n"
        + f"Explained Var:      {explained_variance.sum() * 100:.2f}%\n"
        + f"Explained Var:      {test_explained_var.sum() * 100:.2f}%"
    )

    pca_transform = PCA(n_components=n_components, svd_solver="auto").fit(X_train)
    X_pca = pca_transform.transform(X_train)
    explained_variance = pca_transform.explained_variance_ratio_
    test_explained_var = explain_variance(X_test, pca_transform, X_pca.shape[-1])

    print(
        f"RAW Compression:    {X_pca.shape[-1]} of {X_train.shape[-1]}\n"
        + f"Explained Var:      {explained_variance.sum() * 100:.2f}%\n"
        + f"Explained Var:      {test_explained_var.sum() * 100:.2f}%"
    )
