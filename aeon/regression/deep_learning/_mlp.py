"""Multi Layer Perceptron Network (MLP) for Regression.

__author__ = ["Anonymouscodes911"]
__all__ = ["MLPRegressor"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state

from aeon.regression.deep_learning.base import BaseDeepRegressor

# from aeon.networks import MLPNetwork

class MLPRegressor(BaseDeepRegressor):
    Multi Layer Perceptron Network (MLP) for regression.

    Adapted from the implementation used in :-.

Parameters
----------
n_epochs : int, default = 2000
the number of epochs to train the model
batch_size : int, default = 16
    the number of samples per gradient update.
random_state : int or None, default=None
    Seed for random number generation.
verbose : boolean, default = False
    whether to output extra information
loss : string, default="mean_squared_error"
    fit parameter for the keras model
file_path : str, default = "./"
    file_path when saving model_Checkpoint callback
save_best_model : bool, default = False
    Whether or not to save the best model, if the
    modelcheckpoint callback is used by default,
    this condition, if True, will prevent the
    automatic deletion of the best saved model from
    file and the user can choose the file name
save_last_model : bool, default = False
    Whether or not to save the last model, last
    epoch trained, using the base class method
    save_last_model_to_file
best_file_name : str, default = "best_model"
    The name of the file of the best model, if
    save_best_model is set to False, this parameter
    is discarded
last_file_name : str, default = "last_model"
    The name of the file of the last model, if
    save_last_model is set to False, this parameter
    is discarded
optimizer : keras.optimizer, default=keras.optimizers.Adadelta(),
metrics : list of strings, default=["accuracy"],
activation : string or a tf callable, default="sigmoid"
    Activation function used in the output linear layer.
    List of available activation functions:
    https://keras.io/api/layers/activations/
use_bias : boolean, default = True
    whether the layer uses a bias vector.

Notes
-----
Adapted from the implementation from source code
https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/mlp.py

References
----------
.. #Should we use the same reference from aeon.classification.deep_learning._mlp.py?

Examples
--------
#can we use the similar example here also?
"""
