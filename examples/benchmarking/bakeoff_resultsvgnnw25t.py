# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import uni_classifiers_2017

print(uni_classifiers_2017.keys())


# %%NBQA-CELL-SEPfc780c
from aeon.datasets.tsc_datasets import univariate2015

print(
    f"The {len(univariate2015)} UCR univariate datasets described in [4] and used in "
    f"2017 bakeoff [1]:\n{univariate2015}"
)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import get_bake_off_2017_results

default = get_bake_off_2017_results()
averaged = get_bake_off_2017_results(default_only=False)
print(
    f"{len(univariate2015)} datasets in rows, {len(uni_classifiers_2017)} classifiers "
    f"in columns"
)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_critical_difference

classifiers = ["MSM_1NN", "LPS", "TSBF", "TSF", "DTW_F", "EE", "BOSS", "ST", "FlatCOTE"]
# Get columm positions of classifiers in results
indx = [uni_classifiers_2017[key] for key in classifiers if key in uni_classifiers_2017]
plot, _ = plot_critical_difference(averaged[:, indx], classifiers, test="Nemenyi")
plot.show()


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import multi_classifiers_2021
from aeon.datasets.tsc_datasets import multivariate_equal_length

print(multi_classifiers_2021.keys())
print(
    f"The {len(multivariate_equal_length)} TSML multivariate datasets described in "
    f"and used in the 2021 multivariate bakeoff [1]:\n{multivariate_equal_length}"
)


# %%NBQA-CELL-SEPfc780c
from aeon.benchmarking.results_loaders import get_bake_off_2021_results

default = get_bake_off_2021_results()
averaged = get_bake_off_2021_results(default_only=False)
print("Shape of results = ", averaged.shape)


# %%NBQA-CELL-SEPfc780c
plot, _ = plot_critical_difference(averaged, list(multi_classifiers_2021.keys()))
