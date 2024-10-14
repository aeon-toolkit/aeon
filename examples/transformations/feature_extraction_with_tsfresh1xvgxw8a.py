# %%NBQA-CELL-SEPfc780c
# !pip install --upgrade tsfresh


# %%NBQA-CELL-SEPfc780c
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from aeon.datasets import load_arrow_head, load_basic_motions
from aeon.transformations.collection.feature_based import TSFreshFeatureExtractor

# %%NBQA-CELL-SEPfc780c
X, y = load_arrow_head(return_X_y=True, return_type="nested_univ")
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# %%NBQA-CELL-SEPfc780c
X_train.head()


# %%NBQA-CELL-SEPfc780c
#  binary classification task
np.unique(y_train)


# %%NBQA-CELL-SEPfc780c
# tf = TsFreshTransformer()
t = TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False)
Xt = t.fit_transform(X_train)
Xt.head()


# %%NBQA-CELL-SEPfc780c
classifier = make_pipeline(
    TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False),
    RandomForestClassifier(),
)
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
X, y = load_basic_motions(return_X_y=True, return_type="nested_univ")
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# %%NBQA-CELL-SEPfc780c
#  multivariate input data
X_train.head()


# %%NBQA-CELL-SEPfc780c
t = TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False)
Xt = t.fit_transform(X_train)
Xt.head()


# %%NBQA-CELL-SEPfc780c
from sklearn.ensemble import RandomForestRegressor

from aeon.datasets import load_airline
from aeon.forecasting.base import ForecastingHorizon
from aeon.forecasting.compose import make_reduction
from aeon.forecasting.model_selection import temporal_train_test_split

y = load_airline()
y_train, y_test = temporal_train_test_split(y)

regressor = make_pipeline(
    TSFreshFeatureExtractor(show_warnings=False, disable_progressbar=True),
    RandomForestRegressor(),
)
forecaster = make_reduction(
    regressor, scitype="time-series-regressor", window_length=12
)
forecaster.fit(y_train)

fh = ForecastingHorizon(y_test.index, is_relative=False)
y_pred = forecaster.predict(fh)
