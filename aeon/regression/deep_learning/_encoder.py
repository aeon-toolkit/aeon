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

_tags = {
"python_dependencies": ["tensorflow", "tensorflow_addons"],
}
def __init__(
self,
n_epochs=100,
batch_size=12,
kernel_size=None,
n_filters=None,
dropout_proba=0.2,
activation="sigmoid",
max_pool_size=2,
padding="same",
strides=1,
fc_units=256,
callbacks=None,
file_path="./",
save_best_model=False,
save_last_model=False,
best_file_name="best_model",
last_file_name="last_model",
verbose=False,
loss="categorical_crossentropy",
metrics=None,
random_state=None,
use_bias=True,
optimizer=None,
):
self.n_filters = n_filters
self.max_pool_size = max_pool_size
self.kernel_size = kernel_size
self.strides = strides
self.activation = activation
self.padding = padding
self.dropout_proba = dropout_proba
self.fc_units = fc_units

self.callbacks = callbacks
self.file_path = file_path
self.save_best_model = save_best_model
self.save_last_model = save_last_model
self.best_file_name = best_file_name
self.n_epochs = n_epochs
self.verbose = verbose
self.loss = loss
self.metrics = metrics
self.use_bias = use_bias
self.optimizer = optimizer

self.history = None
super().__init__(
batch_size=batch_size,
random_state=random_state,
last_file_name=last_file_name
)
self._network = EncoderNetwork(
kernel_size=self.kernel_size,
max_pool_size=self.max_pool_size,
n_filters=self.n_filters,
fc_units=self.fc_units,
strides=self.strides,
padding=self.padding,
dropout_proba=self.dropout_proba,
activation=self.activation,
random_state=self.random_state,
)
def build_model(self, input_shape, output_shape, **kwargs):
Construct a compiled, un-trained, keras model that is ready for training.

In aeon, time series are stored in numpy arrays of shape (d, m), where d
is the number of dimensions, m is the series length. Keras/tensorflow assume
data is in shape (m, d). This method also assumes (m, d). Transpose should
happen in fit.

Parameters
----------
input_shape : tuple
The shape of the data fed into the input layer, should be (m, d).
output_shape : int
The number of output units, which becomes the size of the output layer.

Gives
-------
output : a compiled Keras Model
"""

# added a new line for W292
