# %%NBQA-CELL-SEPfc780c
import os

import aeon
from aeon.datasets import load_from_tsfile

DATA_PATH = os.path.join(os.path.dirname(aeon.__file__), "datasets/data")

train_x, train_y = load_from_tsfile(DATA_PATH + "/ArrowHead/ArrowHead_TRAIN.ts")
test_x, test_y = load_from_tsfile(DATA_PATH + "/ArrowHead/ArrowHead_TEST.ts")
test_x[0][0][:5]


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_arrow_head, load_basic_motions, load_plaid

train_x, train_y = load_arrow_head(split="TRAIN")
test_x, test_y = load_arrow_head(split="test")
X, y = load_basic_motions()
plaid_train, _ = load_plaid(split="train")
print("Train shape = ", train_x.shape, " test shape = ", test_x.shape)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_classification

# This will not download, because Arrowhead is already in the directory.
# Change the extract path or name to downloads
X, y, meta_data = load_classification("ArrowHead", return_metadata=True)
print(" Shape of X = ", X.shape)
print(" Meta data = ", meta_data)


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_from_arff_file

X, y = load_from_arff_file(os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.arff"))


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_from_tsv_file

X, y = load_from_tsv_file(os.path.join(DATA_PATH, "ArrowHead/ArrowHead_TRAIN.tsv"))


# %%NBQA-CELL-SEPfc780c
from aeon.datasets import load_from_timeeval_csv_file

AD_DATA_PATH = os.path.join(os.path.dirname(aeon.__file__), "datasets/data")
X, y = load_from_timeeval_csv_file(
    os.path.join(AD_DATA_PATH, "Daphnet_S06R02E0/S06R02E0.csv")
)
X.shape, y.shape
