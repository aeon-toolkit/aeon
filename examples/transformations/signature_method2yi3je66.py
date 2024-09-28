# %%NBQA-CELL-SEPfc780c
# Some additional imports we will use
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from aeon.datasets import load_unit_test


# %%NBQA-CELL-SEPfc780c
# Load an example dataset
train_x, train_y = load_unit_test(split="train")
test_x, test_y = load_unit_test(split="test")


# %%NBQA-CELL-SEPfc780c
from aeon.classification.feature_based import SignatureClassifier
from aeon.transformations.collection.signature_based import SignatureTransformer


# %%NBQA-CELL-SEPfc780c
# First build a very simple signature transform module
signature_transform = SignatureTransformer(
    augmentation_list=("addtime",),
    window_name="global",
    window_depth=None,
    window_length=None,
    window_step=None,
    rescaling=None,
    sig_tfm="signature",
    depth=3,
)

# The simply transform the stream data
print("Raw data shape is: {}".format(train_x.shape))
train_signature_x = signature_transform.fit_transform(train_x)
print("Signature shape is: {}".format(train_signature_x.shape))


# %%NBQA-CELL-SEPfc780c
# Train
model = RandomForestClassifier()
model.fit(train_signature_x, train_y)

# Evaluate
test_signature_x = signature_transform.transform(test_x)
test_pred = model.predict(test_signature_x)
print("Accuracy: {:.3f}%".format(accuracy_score(test_y, test_pred)))


# %%NBQA-CELL-SEPfc780c
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Some params
n_cv_splits = 5
n_gs_iter = 5

# Random forests found to perform very well in general
estimator = RandomForestClassifier()

# The grid to be passed to an sklearn gridsearch
signature_grid = {
    # Signature params
    "depth": [1, 2],
    "window_name": ["dyadic"],
    "augmentation_list": [["basepoint", "addtime"]],
    "window_depth": [1, 2],
    "rescaling": ["post"],
    # Classifier and classifier params
    "estimator": [estimator],
    "estimator__n_estimators": [50, 100],
    "estimator__max_depth": [2, 4],
}

# Initialise the estimator
estimator = SignatureClassifier()

# Run a random grid search and return the gs object
cv = StratifiedKFold(n_splits=n_cv_splits)
gs = RandomizedSearchCV(estimator, signature_grid, cv=n_cv_splits, n_iter=n_gs_iter)
gs.fit(train_x, train_y)

# Get the best classifier
best_classifier = gs.best_estimator_

# Evaluate
train_preds = best_classifier.predict(train_x)
test_preds = best_classifier.predict(test_x)
train_score = accuracy_score(train_y, train_preds)
test_score = accuracy_score(test_y, test_preds)
print(
    "Train acc: {:.3f}%  |  Test acc: {:.3f}%".format(
        train_score * 100, test_score * 100
    )
)
