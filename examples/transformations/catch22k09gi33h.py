# %%NBQA-CELL-SEPfc780c
from sklearn import metrics

from aeon.classification.feature_based import Catch22Classifier
from aeon.datasets import load_basic_motions, load_italy_power_demand
from aeon.transformations.collection.feature_based import Catch22

# %%NBQA-CELL-SEPfc780c
IPD_X_train, IPD_y_train = load_italy_power_demand(split="train")
IPD_X_test, IPD_y_test = load_italy_power_demand(split="test")
IPD_X_test = IPD_X_test[:50]
IPD_y_test = IPD_y_test[:50]

print(IPD_X_train.shape, IPD_y_train.shape, IPD_X_test.shape, IPD_y_test.shape)

BM_X_train, BM_y_train = load_basic_motions(split="train")
BM_X_test, BM_y_test = load_basic_motions(
    split="test",
)

print(BM_X_train.shape, BM_y_train.shape, BM_X_test.shape, BM_y_test.shape)


# %%NBQA-CELL-SEPfc780c
c22_uv = Catch22()
c22_uv.fit(IPD_X_train, IPD_y_train)
transformed_data_uv = c22_uv.transform(IPD_X_train)
print(transformed_data_uv.shape)


# %%NBQA-CELL-SEPfc780c
c22_mv = Catch22()
c22_mv.fit(BM_X_train, BM_y_train)


# %%NBQA-CELL-SEPfc780c
transformed_data_mv = c22_mv.transform(BM_X_train)
print(transformed_data_mv.shape)


# %%NBQA-CELL-SEPfc780c
c22f = Catch22Classifier(random_state=0)
c22f.fit(IPD_X_train, IPD_y_train)


# %%NBQA-CELL-SEPfc780c
c22f_preds = c22f.predict(IPD_X_test)
print("C22F Accuracy: " + str(metrics.accuracy_score(IPD_y_test, c22f_preds)))
