# %%NBQA-CELL-SEPfc780c
# Imports used in this notebook
import numpy as np

from aeon.classification.early_classification._teaser import TEASER
from aeon.classification.interval_based import TimeSeriesForestClassifier
from aeon.datasets import load_arrow_head


# %%NBQA-CELL-SEPfc780c
# Load default train/test splits from aeon/datasets/data
arrow_train_X, arrow_train_y = load_arrow_head(split="train")
arrow_test_X, arrow_test_y = load_arrow_head(split="test")

arrow_test_X.shape


# %%NBQA-CELL-SEPfc780c
teaser = TEASER(
    random_state=0,
    classification_points=[25, 50, 75, 100, 125, 150, 175, 200, 251],
    estimator=TimeSeriesForestClassifier(n_estimators=10, random_state=0),
)
teaser.fit(arrow_train_X, arrow_train_y)


# %%NBQA-CELL-SEPfc780c
hm, acc, earl = teaser.score(arrow_test_X, arrow_test_y)
print("Earliness on Test Data %2.2f" % earl)
print("Accuracy on Test Data %2.2f" % acc)
print("Harmonic Mean on Test Data %2.2f" % hm)


# %%NBQA-CELL-SEPfc780c
print("Earliness on Train Data %2.2f" % teaser._train_earliness)
print("Accuracy on Train Data %2.2f" % teaser._train_accuracy)


# %%NBQA-CELL-SEPfc780c
accuracy = (
    TimeSeriesForestClassifier(n_estimators=10, random_state=0)
    .fit(arrow_train_X, arrow_train_y)
    .score(arrow_test_X, arrow_test_y)
)
print("Accuracy on the full Test Data %2.2f" % accuracy)


# %%NBQA-CELL-SEPfc780c
X = arrow_test_X[:, :, :50]
probas, _ = teaser.predict_proba(X)
idx = (probas >= 0).all(axis=1)
print("First 10 Finished prediction\n", np.argwhere(idx).flatten()[:10])
print("First 10 Probabilities of finished predictions\n", probas[idx][:10])


# %%NBQA-CELL-SEPfc780c
_, acc, _ = teaser.score(X, arrow_test_y)
print("Accuracy with 50 points on Test Data %2.2f" % acc)


# %%NBQA-CELL-SEPfc780c
test_points = [25, 50, 75, 100, 125, 150, 175, 200, 251]
final_states = np.zeros((arrow_test_X.shape[0], 4), dtype=int)
final_decisions = np.zeros(arrow_test_X.shape[0], dtype=int)
open_idx = np.arange(0, arrow_test_X.shape[0])
teaser.reset_state_info()

for i in test_points:
    probas, decisions = teaser.update_predict_proba(arrow_test_X[:, :, :i])
    final_states[open_idx] = teaser.get_state_info()

    arrow_test_X, open_idx, final_idx = teaser.split_indices_and_filter(
        arrow_test_X, open_idx, decisions
    )
    final_decisions[final_idx] = i

    (
        hm,
        acc,
        earliness,
    ) = teaser.compute_harmonic_mean(final_states, arrow_test_y)

    print("Earliness on length %2i is %2.2f" % (i, earliness))
    print("Accuracy on length %2i is %2.2f" % (i, acc))
    print("Harmonic Mean on length %2i is %2.2f" % (i, hm))

    print("...........")

print("Time Stamp of final decisions", final_decisions)
