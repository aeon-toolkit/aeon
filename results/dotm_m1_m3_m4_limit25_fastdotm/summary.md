# DOTM M1/M3/M4 comparison

Series: 652

Paired differences are backend minus baseline. Negative accuracy differences mean the backend is better than the baseline. Negative elapsed differences mean the backend is faster.

## Backend summary

| backend       |   series |   ok |   finite |   total_seconds |   mean_seconds |   median_seconds |   mean_smape |   median_smape |   mean_mase |   median_mase |   mean_mae |   mean_rmse | cold_start_series_key   | cold_start_status   |   cold_start_seconds |   steady_series |   steady_total_seconds |   steady_mean_seconds |   steady_median_seconds |
|:--------------|---------:|-----:|---------:|----------------:|---------------:|-----------------:|-------------:|---------------:|------------:|--------------:|-----------:|------------:|:------------------------|:--------------------|---------------------:|----------------:|-----------------------:|----------------------:|------------------------:|
| aeon          |      652 |  652 |      652 |         3.08653 |     0.00473395 |       0.00259125 |      16.5639 |        10.5648 |     2.58294 |       1.52811 |    26262.6 |     30350.9 | M1:1                    | ok                  |             0.264778 |             651 |                2.82176 |             0.0043345 |               0.0025904 |
| statsforecast |      652 |  652 |      652 |        44.658   |     0.0684939  |       0.0137834  |      16.5662 |        10.5648 |     2.58322 |       1.52875 |    26263.5 |     30351   | M1:1                    | ok                  |            28.45     |             651 |               16.2081  |             0.0248972 |               0.0137703 |

## Forecast and runtime differences

| backend       |   max_abs_forecast_diff |   mean_abs_forecast_diff |   relative_forecast_diff |   time_ratio_backend_over_baseline |
|:--------------|------------------------:|-------------------------:|-------------------------:|-----------------------------------:|
| statsforecast |                 14.9296 |                  8.27743 |              0.000595837 |                            4.85762 |

## Paired tests

| backend       | baseline   | comparison                 |   n |   mean_diff |   median_diff |   ttest_pvalue |   wilcoxon_pvalue |
|:--------------|:-----------|:---------------------------|----:|------------:|--------------:|---------------:|------------------:|
| statsforecast | aeon       | sMAPE backend - baseline   | 652 |  0.00229228 |   4.02141e-07 |       0.594246 |      0.112364     |
| statsforecast | aeon       | MASE backend - baseline    | 652 |  0.00028395 |   4.1393e-08  |       0.391533 |      0.0478382    |
| statsforecast | aeon       | MAE backend - baseline     | 652 |  0.887328   |   7.25141e-06 |       0.59629  |      0.149313     |
| statsforecast | aeon       | RMSE backend - baseline    | 652 |  0.101843   |   5.72142e-06 |       0.971972 |      0.449336     |
| statsforecast | aeon       | seconds backend - baseline | 652 |  0.06376    |   0.010403    |       0.140755 |      8.67663e-106 |

## Difference variation by length

| backend       | length_bin   |   n_series |   smape_diff_mean |   smape_diff_std |   mase_diff_mean |   mase_diff_std |   relative_forecast_diff_mean |   relative_forecast_diff_std |
|:--------------|:-------------|-----------:|------------------:|-----------------:|-----------------:|----------------:|------------------------------:|-----------------------------:|
| statsforecast | 21-38        |        102 |       -0.00572692 |        0.0548129 |     -0.000548955 |      0.00439544 |                   0.000695029 |                  0.00385102  |
| statsforecast | 38-59        |        133 |        0.0192738  |        0.198736  |      0.000272261 |      0.00553055 |                   0.0018655   |                  0.00719489  |
| statsforecast | 59-96        |         87 |        0.00232591 |        0.0102186 |      0.000493546 |      0.00277483 |                   4.88271e-05 |                  0.000183609 |
| statsforecast | 700-2597     |         83 |        0.00225178 |        0.022677  |      0.00302339  |      0.019392   |                   5.87515e-05 |                  0.000241719 |
| statsforecast | 8-21         |        115 |        0.0066273  |        0.0636244 |      0.000130599 |      0.00361135 |                   0.000188189 |                  0.00127267  |
| statsforecast | 96-700       |        132 |       -0.0123947  |        0.115478  |     -0.000787735 |      0.00716536 |                   0.000293296 |                  0.00225757  |

## Difference-length correlations

| backend       | baseline   | difference             |   n_pairs |   pearson_length_correlation |   spearman_length_correlation |    diff_std |    diff_iqr |
|:--------------|:-----------|:-----------------------|----------:|-----------------------------:|------------------------------:|------------:|------------:|
| statsforecast | aeon       | smape_diff             |       652 |                 -0.0123572   |                     0.0441768 |  0.109825   | 7.46145e-06 |
| statsforecast | aeon       | mase_diff              |       652 |                  0.0546301   |                     0.0375848 |  0.00845622 | 1.4742e-06  |
| statsforecast | aeon       | mae_diff               |       652 |                 -0.0121683   |                     0.0507011 | 42.7493     | 0.000396415 |
| statsforecast | aeon       | rmse_diff              |       652 |                 -0.000584572 |                     0.052129  | 73.9852     | 0.000455347 |
| statsforecast | aeon       | relative_forecast_diff |       652 |                 -0.07377     |                     0.339786  |  0.00381758 | 1.77887e-05 |

## Plots

- [smape_vs_length.png](smape_vs_length.png)
- [mase_vs_length.png](mase_vs_length.png)
- [paired_differences_vs_length.png](paired_differences_vs_length.png)

## Pairwise W/D/L

Each row is ``backend_a`` versus ``backend_b`` over the listed subset. ``wins`` = ``backend_a`` strictly beats ``backend_b`` on the metric (lower is better); ``draws`` are within relative tolerance ``1e-06`` or absolute tolerance ``1e-09``; ``losses`` are the converse. Both directions are emitted so each table can be scanned without re-orienting it.

### smape

**dataset: all**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       652 |    133 |     399 |      120 |
| statsforecast | aeon          |       652 |    120 |     399 |      133 |

**dataset: M1**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       151 |     17 |     116 |       18 |
| statsforecast | aeon          |       151 |     18 |     116 |       17 |

**dataset: M3**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       201 |     31 |     146 |       24 |
| statsforecast | aeon          |       201 |     24 |     146 |       31 |

**dataset: M4**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       300 |     85 |     137 |       78 |
| statsforecast | aeon          |       300 |     78 |     137 |       85 |

### mase

**dataset: all**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       652 |    135 |     399 |      118 |
| statsforecast | aeon          |       652 |    118 |     399 |      135 |

**dataset: M1**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       151 |     19 |     116 |       16 |
| statsforecast | aeon          |       151 |     16 |     116 |       19 |

**dataset: M3**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       201 |     31 |     147 |       23 |
| statsforecast | aeon          |       201 |     23 |     147 |       31 |

**dataset: M4**

| backend_a     | backend_b     |   n_pairs |   wins |   draws |   losses |
|:--------------|:--------------|----------:|-------:|--------:|---------:|
| aeon          | statsforecast |       300 |     85 |     136 |       79 |
| statsforecast | aeon          |       300 |     79 |     136 |       85 |
