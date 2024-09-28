# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_classification

X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

print(f"shape of the array: {X_train.shape}")
print(f"n_samples = {X_train.shape[0]}")
print(f"n_channels = {X_train.shape[1]}")
print(f"n_timepoints = {X_train.shape[2]}")


# %%NBQA-CELL-SEPfc780c
length = 25  # the length of the shapelet
S = X_train[0, 0, 45 : 45 + length]  # Set the shapelet values
print(S)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import ShapeletVisualizer

shp_vis = ShapeletVisualizer(S)
fig = shp_vis.plot(figure_options={"figsize": (7, 4)})


# %%NBQA-CELL-SEPfc780c
fig = shp_vis.plot_on_X(X_test[1], figure_options={"figsize": (7, 4)})


# %%NBQA-CELL-SEPfc780c
fig = shp_vis.plot_distance_vector(X_test[1], figure_options={"figsize": (7, 4)})


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

for k, v in all_estimators("transformer", filter_tags={"algorithm_type": "shapelet"}):
    print(f"{k}: {v}")


# %%NBQA-CELL-SEPfc780c
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

st = RandomShapeletTransform(max_shapelets=10).fit(X_train, y_train)
st.transform(X_test).shape


# %%NBQA-CELL-SEPfc780c
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
from aeon.visualisation import ShapeletTransformerVisualizer

st = RandomShapeletTransform(max_shapelets=10).fit(X_train, y_train)
st_vis = ShapeletTransformerVisualizer(st)
id_shapelet = 0  # Identifier of the shapelet

fig = st_vis.plot_on_X(id_shapelet, X_test[1], figure_options={"figsize": (7, 4)})


# %%NBQA-CELL-SEPfc780c
from matplotlib import pyplot as plt

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
st_vis.plot(
    id_shapelet,
    ax=ax[0],
    scatter_options={"c": "purple"},
    line_options={"linestyle": "-."},
)
st_vis.plot_on_X(
    id_shapelet, X_test[1], ax=ax[1], line_options={"linewidth": 3, "alpha": 0.5}
)
ax[1].set_title(f"Best match of shapelet {id_shapelet} on X")
st_vis.plot_distance_vector(
    id_shapelet, X_test[1], ax=ax[2], line_options={"c": "brown"}
)
ax[2].set_title(f"Distance vector of shapelet {id_shapelet} on X")


# %%NBQA-CELL-SEPfc780c
from aeon.registry import all_estimators

all_estimators("classifier", filter_tags={"algorithm_type": "shapelet"})


# %%NBQA-CELL-SEPfc780c
from sklearn.ensemble import RandomForestClassifier

from aeon.classification.shapelet_based import ShapeletTransformClassifier

stc = ShapeletTransformClassifier(estimator=RandomForestClassifier(ccp_alpha=0.01)).fit(
    X_train, y_train
)
stc.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import ShapeletClassifierVisualizer

stc_vis = ShapeletClassifierVisualizer(stc)
id_class = 0
fig = stc_vis.visualize_shapelets_one_class(
    X_test,
    y_test,
    id_class,
    figure_options={"figsize": (18, 12), "nrows": 2, "ncols": 2},
)


# %%NBQA-CELL-SEPfc780c
all_shapelet_classifiers = [
    "MrSQMClassifier",
    "ShapeletTransformClassifier",
    "RDSTClassifier",
    "SASTClassifier",
    "RSASTClassifier",
    "LearningShapeletClassifier",
]
from aeon.benchmarking import get_estimator_results_as_array
from aeon.datasets.tsc_datasets import univariate

est = ["MrSQMClassifier", "RDSTClassifier", "ShapeletTransformClassifier"]
names = [t.replace("Classifier", "") for t in est]
results, present_names = get_estimator_results_as_array(
    names, univariate, include_missing=False
)
results.shape


# %%NBQA-CELL-SEPfc780c
from aeon.visualisation import plot_boxplot, plot_critical_difference

plot_critical_difference(results, names)


# %%NBQA-CELL-SEPfc780c
plot_boxplot(results, names, relative=True)
