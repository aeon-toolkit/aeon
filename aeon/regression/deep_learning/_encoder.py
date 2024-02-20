"""Encoder Regressor.

__author__ = ["AnonymousCodes911"]
__all__ = ["EncoderRegressor"]

import gc
import os
import time
from copy import deepcopy

from sklearn.utils import check_random_state
from aeon.regression.deep_learning.base import BaseDeepRegressor
from aeon.networks import EncoderNetwork

class EncoderRegressor(BaseDeepRegressor):

Establishing the network structure for an Encoder.

Adapted from the implementation used in classification.deeplearning.

Parameters
----------
kernel_size : array of int, default = [5, 11, 21]
    Specifying the length of the 1D convolution windows.
n_filters : array of int, default = [128, 256, 512]
    Specifying the number of 1D convolution filters used for each layer,
    the shape of this array should be the same as kernel_size.
max_pool_size : int, default = 2
    Size of the max pooling windows.
activation : string, default = sigmoid
    Keras activation function.
dropout_proba : float, default = 0.2
    Specifying the dropout layer probability.
padding : string, default = same
    Specifying the type of padding used for the 1D convolution.
strides : int, default = 1
    Specifying the sliding rate of the 1D convolution filter.
fc_units : int, default = 256
    Specifying the number of units in the hidden fully
    connected layer used in the EncoderNetwork.
file_path : str, default = "./"
    File path when saving model_Checkpoint callback.
save_best_model : bool, default = False
    Whether or not to save the best model, if the
    modelcheckpoint callback is used by default,
    this condition, if True, will prevent the
    automatic deletion of the best saved model from
    file and the user can choose the file name.
save_last_model : bool, default = False
    Whether or not to save the last model, last
    epoch trained, using the base class method
    save_last_model_to_file.
best_file_name : str, default = "best_model"
    The name of the file of the best model, if
    save_best_model is set to False, this parameter
    is discarded.
last_file_name : str, default = "last_model"
    The name of the file of the last model, if
    save_last_model is set to False, this parameter
    is discarded.
random_state : int, default = 0
    Seed to any needed random actions.

Notes
-----
Adapted from source code
https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/encoder.py

References
----------
..#need references
"""
