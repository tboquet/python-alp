"""Adaptor for the Keras backend"""

import keras.backend as K
from .utils.keras_utils import model_from_dict_w_opt


def build_predict_func(mod):
    """Build Keras prediction functions based on a Keras model

    Using inputs and outputs of the graph a prediction function
    (forward pass) is compiled for prediction purpose.

    Args:
        mod(keras.models): a Graph or Sequential model

    Returns:
        a Keras (Theano or Tensorflow) function
    """

    return K.function(mod.inputs, mod.outputs, updates=mod.state_updates)


def fit():
    """Dummy fit for now"""
    pass


def  build():
    """Dummy build for now"""
    pass


def predict():
    """Dummy predict for now"""
    pass
