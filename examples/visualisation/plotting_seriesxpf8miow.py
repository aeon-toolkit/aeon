# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_airline, load_arrow_head

airline = load_airline()
airline_train, airline_test = airline[:-24], airline[-24:]
arrowhead_X, arrowhead_y = load_arrow_head()


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import (
    plot_correlations,
    plot_lags,
    plot_series,
    plot_spectrogram,
)


# %%NBQA-CELL-SEPfc780c
_ = plot_series(airline)


# %%NBQA-CELL-SEPfc780c
_ = plot_series(airline_train, airline_test, airline[60:96] + 100)


# %%NBQA-CELL-SEPfc780c
_ = plot_lags(airline, lags=2)


# %%NBQA-CELL-SEPfc780c
_ = plot_lags(airline, lags=[1, 2, 3])


# %%NBQA-CELL-SEPfc780c
_ = plot_correlations(airline)


# %%NBQA-CELL-SEPfc780c
_ = plot_spectrogram(airline)


# %%NBQA-CELL-SEPfc780c
_ = plot_spectrogram(airline, return_onesided=False)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_collection_by_class, plot_series_collection


# %%NBQA-CELL-SEPfc780c
_ = plot_series_collection(arrowhead_X[:3])


# %%NBQA-CELL-SEPfc780c
_ = plot_series_collection(arrowhead_X[:9], arrowhead_y[:9])


# %%NBQA-CELL-SEPfc780c
_ = plot_collection_by_class(arrowhead_X[:9], arrowhead_y[:9])
