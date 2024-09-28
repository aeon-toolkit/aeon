# %%NBQA-CELL-SEPfc780c
import warnings

import matplotlib.pyplot as plt
import numpy as np

from aeon.datasets import load_cardano_sentiment, load_covid_3month

warnings.filterwarnings("ignore")

covid_train, covid_train_y = load_covid_3month(split="train")
covid_test, covid_test_y = load_covid_3month(split="test")
cardano_train, cardano_train_y = load_cardano_sentiment(split="train")
cardano_test, cardano_test_y = load_cardano_sentiment(split="test")
covid_train.shape


# %%NBQA-CELL-SEPfc780c
plt.title("First three cases for Covid3Month")
plt.plot(covid_train[0][0])
plt.plot(covid_train[1][0])
plt.plot(covid_train[2][0])


# %%NBQA-CELL-SEPfc780c
cardano_train.shape


# %%NBQA-CELL-SEPfc780c
plt.title("First three cases for CardanoSentiment")
plt.plot(cardano_train[0][0])
plt.plot(cardano_train[1][0])
plt.plot(cardano_train[2][0])


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

all_estimators("regressor", as_dataframe=True)


# %%NBQA-CELL-SEPfc780c
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor

knn = KNeighborsTimeSeriesRegressor()
knn.fit(covid_train, covid_train_y)
p = knn.predict(covid_test)
sse = np.sum((covid_test_y - p) * (covid_test_y - p))
sse


# %%NBQA-CELL-SEPfc780c
all_estimators(
    "regressor", filter_tags={"capability:multivariate": True}, as_dataframe=True
)


# %%NBQA-CELL-SEPfc780c
from aeon.regression import DummyRegressor

fp = KNeighborsTimeSeriesRegressor()
dummy = DummyRegressor()
dummy.fit(cardano_train, cardano_train_y)
knn.fit(cardano_train, cardano_train_y)
pred = knn.predict(cardano_test)
res_knn = cardano_test_y - pred
res_dummy = cardano_test_y - dummy.predict(cardano_test)
plt.title("Raw residuals")
plt.plot(res_knn)
plt.plot(res_dummy)
