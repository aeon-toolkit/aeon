# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_airline, load_electric_devices_segmentation

ed_seg, _, ed_seg_chp = load_electric_devices_segmentation()
airline = load_airline()


# %%NBQA-CELL-SEPfc780c
from aeon.forecasting.model_selection import SlidingWindowSplitter
from aeon.visualisation import plot_series_windows

# %%NBQA-CELL-SEPfc780c
fh = list(range(1, 13))
_ = plot_series_windows(
    airline, SlidingWindowSplitter(fh=fh, window_length=60, step_length=len(fh))
)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_series_with_change_points

# %%NBQA-CELL-SEPfc780c
_ = plot_series_with_change_points(ed_seg, ed_seg_chp)
