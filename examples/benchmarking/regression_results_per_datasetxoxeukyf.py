# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_covid_3month  # univariate regression dataset
from aeon.regression.feature_based import FreshPRINCERegressor
from aeon.regression.interval_based import DrCIFRegressor
from aeon.visualisation import plot_scatter_predictions

X_train, y_train = load_covid_3month(split="train")
X_test, y_test = load_covid_3month(split="test")

# Running FP
fp = FreshPRINCERegressor(n_estimators=10, default_fc_parameters="minimal")
fp.fit(X_train, y_train)
y_pred_fp = fp.predict(X_test)

# Running DrCIF
drcif = DrCIFRegressor(n_estimators=10)
drcif.fit(X_train, y_train)
y_pred_drcif = drcif.predict(X_test)


# %%NBQA-CELL-SEPfc780c
fig, ax = plot_scatter_predictions(y_test, y_pred_fp, title="FreshPRINCE - Covid3Month")

fig.show()


# %%NBQA-CELL-SEPfc780c
fig, ax = plot_scatter_predictions(y_test, y_pred_drcif, title="DrCIF - Covid3Month")

fig.show()
