"""Multi Layer Perceptron Network (MLP) for Regression.

 __author__ = ["Anonymouscodes911"]
 __all__ = ["MLPRegressor"]


import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.classification.deep_learning.base import BaseDeepRegressor

# from aeon.networks import MLPNetwork
"""

# Added a new line at the end to handle flake8 error(W292)
