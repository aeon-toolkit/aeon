"""Example usage of ESMOTE and ESMOTE_SOFT.

This script is intended as a lightweight smoke test and usage example for the
standard MSM e-SMOTE transformer and the Soft-MSM e-SMOTE extension.
"""

import numpy as np

from aeon.datasets import load_classification
from aeon.transformations.collection.imbalance._esmote import ESMOTE
from aeon.transformations.collection.imbalance._esmote_soft import ESMOTE_SOFT


def make_imbalanced_binary_problem(X, y, minority_fraction=0.2, random_state=0):
    """Create a simple binary imbalanced problem for testing.

    This is only for demonstration. In the real experiments we should use the
    fixed imbalanced archive from the e-SMOTE / MGD-CVAE papers.
    """
    rng = np.random.default_rng(random_state)

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("Need at least two classes to create a binary problem.")

    # Keep only two classes.
    class_a, class_b = classes[:2]
    mask = np.isin(y, [class_a, class_b])
    X = X[mask]
    y = y[mask]

    # Relabel to 0/1 for convenience.
    y = np.where(y == class_a, 0, 1)

    # Make class 1 the minority by downsampling it.
    majority_idx = np.flatnonzero(y == 0)
    minority_idx = np.flatnonzero(y == 1)

    n_minority = max(2, int(len(majority_idx) * minority_fraction))
    n_minority = min(n_minority, len(minority_idx))

    selected_minority_idx = rng.choice(minority_idx, size=n_minority, replace=False)

    selected_idx = np.concatenate([majority_idx, selected_minority_idx])
    rng.shuffle(selected_idx)

    return X[selected_idx], y[selected_idx]


def print_dataset_summary(title, X, y):
    """Print shape and class counts."""
    print(f"\n{title}")  # noqa: T201
    print("X shape:", X.shape)  # noqa: T201
    print("class counts:", dict(zip(*np.unique(y, return_counts=True))))  # noqa: T201


def main():
    """Run small ESMOTE and Soft-MSM e-SMOTE examples."""
    # Load a tiny aeon example dataset.
    # Shape is normally (n_cases, n_channels, n_timepoints).
    X, y = load_classification("GunPoint")

    X_imb, y_imb = make_imbalanced_binary_problem(
        X, y, minority_fraction=0.2, random_state=0
    )

    print_dataset_summary("Before rebalancing", X_imb, y_imb)

    # First, run the existing MSM e-SMOTE implementation.
    # This is the baseline we want to compare the Soft-MSM extension against.
    esmote = ESMOTE(
        n_neighbors=1,
        random_state=0,
        distance="msm",
    )

    X_esmote, y_esmote = esmote.fit_transform(X_imb, y_imb)

    print_dataset_summary("After standard MSM e-SMOTE rebalancing", X_esmote, y_esmote)

    # Main Soft-MSM version for the TKDE proof-of-concept:
    # neighbour search uses Soft-MSM, and synthetic generation uses the
    # Soft-MSM alignment matrix to form an expected aligned counterpart.
    soft_resampler = ESMOTE_SOFT(
        k_neighbors=1,
        distance="soft_msm",
        generation="soft_path",
        msm_cost=1.0,
        gamma=1.0,
        window=None,
        itakura_max_slope=None,
        random_state=0,
    )

    X_soft, y_soft = soft_resampler.fit_transform(X_imb, y_imb)

    print_dataset_summary("After Soft-MSM e-SMOTE rebalancing", X_soft, y_soft)

    # Optional slower variant:
    # This generates each synthetic as an approximate two-series Soft-MSM
    # barycentre:
    #
    #     z* = argmin_z (1-lambda) SoftMSM(z, x)
    #                  + lambda     SoftMSM(z, y)
    #
    # Use this only after soft_path is working.
    bary_resampler = ESMOTE_SOFT(
        k_neighbors=3,
        distance="soft_msm",
        generation="soft_barycenter",
        msm_cost=1.0,
        gamma=1.0,
        max_iter=25,
        learning_rate=0.05,
        tol=1e-6,
        random_state=0,
    )

    X_bary, y_bary = bary_resampler.fit_transform(X_imb, y_imb)

    print_dataset_summary(
        "After Soft-MSM barycentre e-SMOTE rebalancing", X_bary, y_bary
    )


if __name__ == "__main__":
    main()
