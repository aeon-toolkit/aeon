"""Code to select datasets for regression-based forecasting experiments."""

import gc
import os
import tempfile
import time

import pandas as pd

from aeon.datasets import load_forecasting
from aeon.datasets._data_writers import (
    write_forecasting_dataset,
    write_regression_dataset,
)

filtered_datasets = [
    "nn5_daily_dataset_without_missing_values",
    "nn5_weekly_dataset",
    "m1_yearly_dataset",
    "m1_quarterly_dataset",
    "m1_monthly_dataset",
    "m3_yearly_dataset",
    "m3_quarterly_dataset",
    "m3_monthly_dataset",
    "m3_other_dataset",
    "m4_yearly_dataset",
    "m4_quarterly_dataset",
    "m4_monthly_dataset",
    "m4_weekly_dataset",
    "m4_daily_dataset",
    "m4_hourly_dataset",
    "tourism_yearly_dataset",
    "tourism_quarterly_dataset",
    "tourism_monthly_dataset",
    "car_parts_dataset_without_missing_values",
    "hospital_dataset",
    "weather_dataset",
    "dominick_dataset",
    "fred_md_dataset",
    "solar_10_minutes_dataset",
    "solar_weekly_dataset",
    "solar_4_seconds_dataset",
    "wind_4_seconds_dataset",
    "sunspot_dataset_without_missing_values",
    "wind_farms_minutely_dataset_without_missing_values",
    "elecdemand_dataset",
    "us_births_dataset",
    "saugeenday_dataset",
    "covid_deaths_dataset",
    "cif_2016_dataset",
    "london_smart_meters_dataset_without_missing_values",
    "kaggle_web_traffic_dataset_without_missing_values",
    "kaggle_web_traffic_weekly_dataset",
    "traffic_hourly_dataset",
    "traffic_weekly_dataset",
    "electricity_hourly_dataset",
    "electricity_weekly_dataset",
    "pedestrian_counts_dataset",
    "kdd_cup_2018_dataset_without_missing_values",
    "australian_electricity_demand_dataset",
    "covid_mobility_dataset_without_missing_values",
    "rideshare_dataset_without_missing_values",
    "vehicle_trips_dataset_without_missing_values",
    "temperature_rain_dataset_without_missing_values",
    "oikolab_weather_dataset",
]


def filter_datasets():
    """
    Filter datasets to identify and print time series with more than 1000 data points.

    This function iterates over a list of datasets, loads each dataset,
    and checks each time series within it. If a series contains more than 1000
    data points, it is counted as a "hit." The function prints up to 10 matches
    per dataset in the format: `<dataset_name>,<series_name>`.

    Returns
    -------
    None
        The function does not return anything but prints matching dataset
        and series names to the console.

    Notes
    -----
    - The function introduces a 1-second delay (`time.sleep(1)`) between processing
      datasets to control HTTP request frequency.
    - Uses `gc.collect()` to explicitly trigger garbage collection, to avoid
      running out of memory
    """
    num_hits = 0
    for dataset_name in filtered_datasets:
        # print(f"{dataset_name}")
        time.sleep(1)
        dataset_counter = 0
        dataset = load_forecasting(dataset_name)
        for index, row in enumerate(dataset["series_value"]):
            if len(row) > 1000:
                num_hits += 1
                dataset_counter += 1
                if dataset_counter <= 10:
                    print(f"{dataset_name},{dataset['series_name'][index]}")  # noqa
        # if dataset_counter > 0:
        #     print(f"{dataset_name}: Hits: {dataset_counter}")
        del dataset
        gc.collect()
    # print(f"Num hits in datasets: {num_hits}")


# filter_datasets()


def filter_and_categorise_m4(frequency_type):
    """
    Filter and categorize M4 dataset time series.

    Parameters
    ----------
    frequency_type : str
        The frequency type of the M4 dataset to process.
        Accepted values: 'yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly'.

    Returns
    -------
    None
        The function does not return any values but prints categorized series
        information.

    Notes
    -----
    - The function constructs an appropriate prefix ('Y', 'Q', 'M', 'W', 'D', 'H')
    based on the dataset type to match metadata identifiers.
    - Limits printed results to 10 per category.
    """
    metadata = pd.read_csv("C:/Users/alexb/Downloads/M4-info.csv")
    m4daily = load_forecasting(f"m4_{frequency_type}_dataset")
    categories = {}
    prefix = ""
    if frequency_type == "yearly":
        prefix = "Y"
    elif frequency_type == "quarterly":
        prefix = "Q"
    elif frequency_type == "monthly":
        prefix = "M"
    elif frequency_type == "weekly":
        prefix = "W"
    elif frequency_type == "daily":
        prefix = "D"
    elif frequency_type == "hourly":
        prefix = "H"
    for index, row in enumerate(m4daily["series_value"]):
        if len(row) > 1000:
            category = metadata.loc[
                metadata["M4id"] == f"{prefix}{m4daily['series_name'][index][1:]}",
                "category",
            ].values[0]
            if category not in categories:
                categories[category] = 1
            else:
                categories[category] += 1
            if categories[category] <= 10:
                print(  # noqa
                    f"m4_{frequency_type}_dataset,\
                    {m4daily['series_name'][index]},{category}"
                )


# filter_and_categorise_m4('monthly')
# filter_and_categorise_m4('weekly')
# filter_and_categorise_m4('daily')
# filter_and_categorise_m4('hourly')


def gen_datasets(problem_type):
    """
    Generate windowed train/test split of datasets.

    Returns
    -------
    None
        The function does not return anything but writes out the train and test
        files to the specified directory.

    Notes
    -----
    - Requires a CSV file containing a list of the series to process.
    """
    final_series_selection = pd.read_csv(
        "./aeon/datasets/forecasting/Final Dataset Selection.csv"
    )
    current_dataset = ""
    dataset = pd.DataFrame()
    tmpdir = tempfile.mkdtemp()
    location_of_datasets = f"./aeon/datasets/local_data/{problem_type}"
    if not os.path.exists(location_of_datasets):
        os.makedirs(location_of_datasets)
    with open(f"{location_of_datasets}/windowed_series.txt", "w") as f:
        for item in final_series_selection.to_records(index=False):
            if current_dataset != item[0]:
                dataset = load_forecasting(item[0], tmpdir)
                current_dataset = item[0]
                print(f"Current Dataset: {current_dataset}")  # noqa
            f.write(f"{item[0]}_{item[1]}\n")
            series = (
                dataset[dataset["series_name"] == item[1]]["series_value"]
                .iloc[0]
                .to_numpy()
            )
            dataset_name = f"{item[0]}_{item[1]}"
            full_file_path = f"{location_of_datasets}/{dataset_name}"
            if not os.path.exists(full_file_path):
                os.makedirs(full_file_path)
            if problem_type == "regression":
                write_regression_dataset(series, full_file_path, dataset_name)
            elif problem_type == "forecasting":
                write_forecasting_dataset(series, full_file_path, dataset_name)


gen_datasets("regression")
