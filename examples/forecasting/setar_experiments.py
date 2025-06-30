"""Temporal file to do setar tree related experiments."""

import os

import numpy as np
import requests

from aeon.datasets import load_from_tsf_file
from aeon.forecasting._setartree import SetartreeForecaster


def download_file(url, local_filename):
    """Download a file from a URL and saves it locally."""
    if not os.path.exists(local_filename):
        # print(f"Downloading {local_filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # print("Download complete.")
    # else:
    # print(f"{local_filename} already exists.")
    return local_filename


def calculate_msmape(y_true, y_pred):
    """Calculate the modified Symmetric Mean Absolute Percentage Error (msMAPE)."""
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Epsilon as defined in the paper's formula
    epsilon = 0.1

    numerator = np.abs(y_pred - y_true)
    denominator = (
        np.maximum(np.abs(y_true) + np.abs(y_pred) + epsilon, 0.5 + epsilon)
    ) / 2.0

    # Calculate msMAPE for each series
    per_series_msmape = numerator / denominator

    # Return the mean msMAPE across all series
    return np.mean(per_series_msmape) * 100


def calculate_mase(y_true, y_pred, y_train, seasonality=1):
    """Calculate the Mean Absolute Scaled Error (MASE)."""
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Calculate the Mean Absolute Error on the test set for each series
    mae_test = np.mean(np.abs(y_true - y_pred), axis=1)

    # Calculate the MAE for the naive seasonal forecast on the training set
    naive_forecast_errors = np.abs(y_train[:, seasonality:] - y_train[:, :-seasonality])
    mae_naive_train = np.mean(naive_forecast_errors, axis=1)

    # To avoid division by zero for flat series, replace 0s with a small number
    mae_naive_train[mae_naive_train == 0] = np.finfo(np.float64).eps

    # Calculate MASE for each series
    per_series_mase = mae_test / mae_naive_train

    # Return the mean MASE across all series
    return np.mean(per_series_mase)


# 1. Load the dataset
# print("Loading dataset...")
base_url = "https://github.com/rakshitha123/SETAR_Trees/raw/master/datasets/"
dataset_name = "chaotic_logistic_dataset.tsf"
url = base_url + dataset_name
local_filename = "chaotic_logistic_dataset.tsf"
download_file(url, local_filename)

loaded_data, metadata = load_from_tsf_file(local_filename)

# 2. Convert to NumPy array
y_list = loaded_data["series_value"].tolist()
y = np.array(y_list, dtype=np.float64)
# print(f"Dataset shape: {y.shape}")

# 3. Split the data
forecast_horizon = 8
y_train = y[:, :-forecast_horizon]
y_test = y[:, -forecast_horizon:]

# print(f"Training data shape: {y_train.shape}")
# print(f"Test data shape: {y_test.shape}")

# 4. Instantiate and fit the forecaster
forecaster = SetartreeForecaster(lag=10, horizon=forecast_horizon)

# print("\nFitting the SetartreeForecaster...")
y_fit = y_train[0]
exog_fit = y_train[1:] if y_train.shape[0] > 1 else None
# exog_fit = y_train[1:10] if y_train.shape[0] > 1 else None

forecaster.fit(y_fit, exog=exog_fit)
# print("Fitting complete.")

# 5. Make predictions
# print("\nMaking predictions for each series...")
all_predictions = []
for i in range(len(y_train)):
    history = y_train[i]
    prediction = forecaster.predict(y=history)
    all_predictions.append(prediction)

y_pred = np.array(all_predictions)
# print("Prediction complete.")

# 6. Evaluate the forecast using msMAPE and MASE
# We compare the single predicted step against the final actual step
y_test_final_step = y_test[:, -1]
# y_test_final_step = y_test[0:10, -1]

# Reshape for metric functions which expect 2D arrays
y_pred_reshaped = y_pred.reshape(-1, 1)
y_test_final_reshaped = y_test_final_step.reshape(-1, 1)


msmape = calculate_msmape(y_test_final_reshaped, y_pred_reshaped)
# For the chaotic logistic dataset, the paper considers seasonality S=1
mase = calculate_mase(y_test_final_reshaped, y_pred_reshaped, y_train, seasonality=1)

# print(f"\nModified Symmetric Mean Absolute Percentage Error (msMAPE): {msmape:.4f}%")
# print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")

# Example of a single series forecast vs actual
# print("\nExample Forecast (final step only):")
# print("Actual value (first series, final step):", np.round(y_test[0, -1], 2))
# print("Predicted value (first series, final step):", np.round(y_pred[0], 2))
