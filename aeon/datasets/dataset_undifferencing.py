"""Undifference Forecasts."""

import os

import pandas as pd

from aeon.datasets import load_from_ts_file
from aeon.transformations.series import DifferenceTransformer

FORECASTING_ALGORITHMS = [
    "ETSForecaster",
    "AutoETSForecaster",
    "AutoARIMA",
    "AutoSARIMA",
    # "SktimeETS",
    # "StatsForecastETS",
    "NaiveForecaster",
]

ALGORITHMS = [
    "RocketRegressor",
    "MultiRocketRegressor",
    "ResNetRegressor",
    "fpcregressor",
    "fpcr-b-spline",
    "TimeCNNRegressor",
    "FCNRegressor",
    "1nn-ed",
    "1nn-dtw",
    "5nn-ed",
    "5nn-dtw",
    "FreshPRINCERegressor",
    "TimeSeriesForestRegressor",
    "DrCIFRegressor",
    "Ridge",
    "SVR",
    "RandomForestRegressor",
    "RotationForestRegressor",
    "xgboost",
    "IndividualInceptionRegressor",
    "InceptionTimeRegressor",
]


def recover_forecasts(
    location_of_datasets, location_of_results, run_name, forecasting=False
):
    """
    Load differenced predictions from CSV and reconstruct the original forecasts.

    Parameters
    ----------
    predictions_csv_path : str
        Path to the CSV file with differenced predictions.
    original_data : np.ndarray
        The original undifferenced time series.
    order : int
        The differencing order used in forecasting.

    Returns
    -------
    np.ndarray
        The recovered forecast values.
    """
    with open(
        f"{location_of_datasets}/windowed_series.txt", encoding="utf-8"
    ) as windowtxt:
        series_list = windowtxt.readlines()
        algorithms = FORECASTING_ALGORITHMS if forecasting else ALGORITHMS
        for algorithm in algorithms:
            for series in series_list:
                series = series.strip()
                if forecasting:
                    original_data = pd.read_csv(
                        f"{location_of_datasets}/{series}/{series}_TEST.csv",
                        header=None,
                    ).values.squeeze()
                else:
                    original_data = load_from_ts_file(
                        f"{location_of_datasets}/{series}/{series}_TEST.ts"
                    )[1]
                predictions_csv_path = f"{location_of_results}/{run_name}/{algorithm}\
                    /Predictions/{series}/testResample0.csv"
                location_of_undifferenced = f"{location_of_results}/\
                    {run_name}_recovered/{algorithm}/Predictions/{series}"
                if not os.path.exists(predictions_csv_path):
                    print(f"Skipping: {algorithm}/{series}")  # noqa
                    continue
                if os.path.exists(f"{location_of_undifferenced}/testResample0.csv"):
                    continue  # Skip if already processed
                # Read all lines including metadata
                with open(predictions_csv_path, encoding="utf-8") as f:
                    lines = f.readlines()

                # Read predictions starting from line 4 (index 3)
                df = pd.read_csv(predictions_csv_path, skiprows=3, header=None)

                # Extract differenced predictions (assumed in column 1)
                differenced_preds = df.iloc[:, 1].values

                # Use inverse transform to reconstruct the original forecasts
                recovered_forecasts = (
                    DifferenceTransformer()
                    .fit(differenced_preds)
                    .inverse_transform(differenced_preds, original_data)
                )

                # Apply undifferencing on column 1
                df.iloc[:, 1] = recovered_forecasts
                df.iloc[:, 0] = original_data[
                    : len(recovered_forecasts)
                ]  # Ensure the first column has original data
                # Convert dataframe back to CSV format lines (excluding metadata)
                forecast_lines = df.to_csv(index=False, header=False).splitlines(
                    keepends=True
                )

                # Write metadata + updated forecast rows to new file
                if not os.path.exists(location_of_undifferenced):
                    os.makedirs(location_of_undifferenced)
                with open(
                    f"{location_of_undifferenced}/testResample0.csv",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.writelines(lines[:3])  # Copy metadata/header
                    f.writelines(forecast_lines)  # Write modified forecasts


recover_forecasts(
    "./aeon/datasets/local_data/regression",
    "../ForecastingResults",
    "part_diff_regression",
    False,
)
