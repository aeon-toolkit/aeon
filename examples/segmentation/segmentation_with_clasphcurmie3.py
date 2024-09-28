# %%NBQA-CELL-SEPfc780c
import sys

sys.path.insert(0, "..")

import pandas as pd
import seaborn as sns

sns.set_theme()
sns.set_color_codes()

from aeon.datasets import load_electric_devices_segmentation
from aeon.segmentation import ClaSPSegmenter, find_dominant_window_sizes
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles


# %%NBQA-CELL-SEPfc780c
ts, period_size, true_cps = load_electric_devices_segmentation()
_ = plot_series_with_change_points(ts, true_cps, title="Electric Devices")


# %%NBQA-CELL-SEPfc780c
# ts is a pd.Series
# we convert it into a DataFrame for display purposed only
pd.DataFrame(ts)


# %%NBQA-CELL-SEPfc780c
clasp = ClaSPSegmenter(period_length=period_size, n_cps=5)
found_cps = clasp.fit_predict(ts)
profiles = clasp.profiles
scores = clasp.scores
print("The found change points are", found_cps)


# %%NBQA-CELL-SEPfc780c
_ = plot_series_with_profiles(
    ts,
    profiles,
    true_cps=true_cps,
    found_cps=found_cps,
    title="Electric Devices",
)


# %%NBQA-CELL-SEPfc780c
clasp = ClaSPSegmenter(period_length=period_size, n_cps=5)
found_segmentation = clasp.fit_predict(ts)
print(found_segmentation)


# %%NBQA-CELL-SEPfc780c
dominant_period_size = find_dominant_window_sizes(ts)
print("Dominant Period", dominant_period_size)


# %%NBQA-CELL-SEPfc780c
clasp = ClaSPSegmenter(period_length=dominant_period_size, n_cps=5)
found_cps = clasp.fit_predict(ts)
profiles = clasp.profiles
scores = clasp.scores

_ = plot_series_with_profiles(
    ts,
    profiles,
    true_cps=true_cps,
    found_cps=found_cps,
    title="ElectricDevices",
)
