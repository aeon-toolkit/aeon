# %%NBQA-CELL-SEPfc780c
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from aeon.datasets import load_basic_motions
from aeon.transformations.collection import channel_selection
from aeon.transformations.collection.convolution_based import Rocket

X_train, y_train = load_basic_motions(split="train")
X_test, y_test = load_basic_motions(split="test")
X_train.shape, X_test.shape


# %%NBQA-CELL-SEPfc780c
# cs = channel_selection.ElbowClassSum()  # ECS
cs = channel_selection.ElbowClassPairwise(prototype_type="mad")  # ECP
rocket_pipeline = make_pipeline(cs, Rocket(), RidgeClassifierCV())


# %%NBQA-CELL-SEPfc780c
rocket_pipeline.fit(X_train, y_train)
rocket_pipeline.score(X_test, y_test)


# %%NBQA-CELL-SEPfc780c
X_selected = cs.fit(X_train, y_train)
cs.channels_selected_


# %%NBQA-CELL-SEPfc780c
cs.distance_frame
