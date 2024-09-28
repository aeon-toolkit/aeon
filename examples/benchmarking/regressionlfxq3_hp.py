# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_cardano_sentiment, load_covid_3month, load_regression

trainX, trainy = load_covid_3month(split="train")
testX, testy = load_regression(split="test", name="Covid3Month")
X, y = load_cardano_sentiment()  # Combines train and test splits
print(trainX.shape, testX.shape, X.shape)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.tser_datasets import tser_soton

print(sorted(list(tser_soton)))


# %%NBQA-CELL-SEPfc780c
small_problems = [
    "CardanoSentiment",
    "Covid3Month",
]

for problem in small_problems:
    X, y = load_regression(name=problem)
    print(problem, X.shape, y.shape)


# %%NBQA-CELL-SEPfc780c
for problem in small_problems:
    trainX, trainy = load_regression(name=problem, split="train")
    print(problem, X.shape, y.shape)


# %%NBQA-CELL-SEPfc780c
from sklearn.metrics import mean_squared_error

from aeon.regression import DummyRegressor

dummy = DummyRegressor()
performance = []
for problem in small_problems:
    trainX, trainy = load_regression(name=problem, split="train")
    dummy.fit(trainX, trainy)
    testX, testy = load_regression(name=problem, split="test")
    predictions = dummy.predict(testX)
    mse = mean_squared_error(testy, predictions)
    performance.append(mse)
    print(problem, " Dummy score = ", mse)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_available_estimators, get_estimator_results

print(get_available_estimators(task="regression"))
results = get_estimator_results(
    estimators=["DrCIF", "FreshPRINCE"],
    task="regression",
    datasets=small_problems,
    measure="mse",
)
print(results)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array

results, names = get_estimator_results_as_array(
    estimators=["DrCIF", "FreshPRINCE"],
    task="regression",
    datasets=small_problems,
    measure="mse",
)
print(results)
print(names)


# %%NBQA-CELL-SEPfc780c
import numpy as np

paired_sorted = sorted(zip(names, results))
names, _ = zip(*paired_sorted)
sorted_rows = [row for _, row in paired_sorted]
sorted_results = np.array(sorted_rows)
print(names)
print(sorted_results)


# %%NBQA-CELL-SEPfc780c
paired = sorted(zip(small_problems, performance))
small_problems, performance = zip(*paired)
print(small_problems)
print(performance)
all_results = np.column_stack((sorted_results, performance))
print(all_results)
regressors = ["DrCIF", "FreshPRINCE", "Dummy"]


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_pairwise_scatter

fig, ax = plot_pairwise_scatter(
    all_results[:, 1],
    all_results[:, 2],
    "FreshPRINCE",
    "Dummy",
    metric="mse",
    lower_better=True,
)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_critical_difference

res = plot_critical_difference(
    all_results,
    regressors,
    lower_better=True,
)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot

res = plot_boxplot(
    all_results,
    regressors,
    relative=True,
)
