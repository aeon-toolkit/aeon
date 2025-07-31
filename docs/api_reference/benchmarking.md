# Benchmarking

The `aeon.benchmarking` module contains tools for comparing and evaluating time
series models, loading stored results, and calculating performance metrics for a
variety of tasks.

## Results loading

Results loaders and loading utilities for `aeon` (and other) estimators.

```{eval-rst}
.. currentmodule:: aeon.benchmarking.results_loaders

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    estimator_alias
    get_available_estimators
    get_estimator_results
    get_estimator_results_as_array
```

## Published results

Results loaders for specific publications.

```{eval-rst}
.. currentmodule:: aeon.benchmarking.published_results

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_classification_bake_off_2017_results
    load_classification_bake_off_2021_results
    load_classification_bake_off_2023_results
```

## Resampling

Functions for resampling time series data.

```{eval-rst}
.. currentmodule:: aeon.benchmarking.resampling

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    resample_data
    resample_data_indices
    stratified_resample_data
    stratified_resample_data_indices
```

## Performance metrics

Performance metrics used for evaluating `aeon` estimators.

### Anomaly Detection

```{eval-rst}
.. currentmodule:: aeon.benchmarking.metrics.anomaly_detection

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    range_precision
    range_recall
    range_f_score
    roc_auc_score
    pr_auc_score
    rp_rr_auc_score
    f_score_at_k_points
    f_score_at_k_ranges
    range_pr_roc_auc_support
    range_roc_auc_score
    range_pr_auc_score
    range_pr_vus_score
    range_roc_vus_score
```

### Anomaly detection thresholding

```{eval-rst}
.. currentmodule:: aeon.benchmarking.metrics.anomaly_detection.thresholding

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    percentile_threshold
    sigma_threshold
    top_k_points_threshold
    top_k_ranges_threshold
```

### Clustering

```{eval-rst}
.. currentmodule:: aeon.benchmarking.metrics.clustering

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    clustering_accuracy_score
```

### Segmentation

```{eval-rst}
.. currentmodule:: aeon.benchmarking.metrics.segmentation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    count_error
    hausdorff_error
    prediction_ratio
```

## Stats

```{eval-rst}
.. currentmodule:: aeon.benchmarking.stats

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_friedman
    nemenyi_test
    wilcoxon_test
```
