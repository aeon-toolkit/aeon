# %%NBQA-CELL-SEPfc780c
import warnings

from sklearn import metrics

from aeon.classification.dictionary_based import (
    MUSE,
    WEASEL,
    BOSSEnsemble,
    ContractableBOSS,
    IndividualBOSS,
    TemporalDictionaryEnsemble,
)
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.registry import all_estimators

warnings.filterwarnings("ignore")
all_estimators(
    "classifier", filter_tags={"algorithm_type": "dictionary"}, as_dataframe=True
)


# %%NBQA-CELL-SEPfc780c
X_train, y_train = load_italy_power_demand(split="train")
X_test, y_test = load_italy_power_demand(split="test")
X_test = X_test[:50]
y_test = y_test[:50]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

X_train_mv, y_train_mv = load_basic_motions(split="train")
X_test_mv, y_test_mv = load_basic_motions(split="test")

X_train_mv = X_train_mv[:20]
y_train_mv = y_train_mv[:20]
X_test_mv = X_test_mv[:20]
y_test_mv = y_test_mv[:20]

print(X_train_mv.shape, y_train_mv.shape, X_test_mv.shape, y_test_mv.shape)


# %%NBQA-CELL-SEPfc780c
one_boss = IndividualBOSS(window_size=8, word_length=4, alphabet_size=6)
boss = BOSSEnsemble(random_state=47)
boss.fit(X_train, y_train)

boss_preds = boss.predict(X_test)
print("BOSS Accuracy: " + str(metrics.accuracy_score(y_test, boss_preds)))
cboss = ContractableBOSS(n_parameter_samples=250, max_ensemble_size=50, random_state=47)
cboss.fit(X_train, y_train)

cboss_preds = cboss.predict(X_test)
print("cBOSS Accuracy: " + str(metrics.accuracy_score(y_test, cboss_preds)))


# %%NBQA-CELL-SEPfc780c
weasel = WEASEL(binning_strategy="equi-depth", anova=False, random_state=47)
weasel.fit(X_train, y_train)

weasel_preds = weasel.predict(X_test)
print(
    f"Univariate WEASEL Accuracy on ItalyPowerDemand: "
    f"{metrics.accuracy_score(y_test, weasel_preds)}"
)

muse = MUSE()
muse.fit(X_train_mv, y_train_mv)

muse_preds = muse.predict(X_test_mv)
print(
    f"Multivariate MUSE Accuracy on BasicMotions: "
    f"{metrics.accuracy_score(y_test_mv, muse_preds)}"
)


# %%NBQA-CELL-SEPfc780c
# Recommended non-contract TDE parameters
tde = TemporalDictionaryEnsemble(
    n_parameter_samples=250,
    max_ensemble_size=50,
    randomly_selected_params=50,
    random_state=47,
)

# If you wish to set a time contract to, for example, 5 minutes,
# set time_limit_in_minutes = 5 in the constructor
# Univariate
tde.fit(X_train, y_train)

tde_preds = tde.predict(X_test)
print(
    "TDE Accuracy on ItalyPowerDemand: "
    + str(metrics.accuracy_score(y_test, tde_preds))
)
tde.fit(X_train_mv, y_train_mv)

tde_preds = tde.predict(X_test_mv)
print(
    f"TDE Accuracy on BasicMotions: " f"{metrics.accuracy_score(y_test_mv, tde_preds)}"
)


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

est = all_estimators("classifier", filter_tags={"algorithm_type": "dictionary"})
for c in est:
    print(c)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

names = [t[0] for t in est]
names.remove("MUSE")  # Multivariate classifier
names.remove("OrdinalTDE")  # Ordinal classifier
names.remove("REDCOMETS")  # We still need to evaluate this classifier

results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
