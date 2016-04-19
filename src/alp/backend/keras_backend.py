"""Adaptor for the Keras backend"""

import keras.backend as K
from .utils.keras_utils import model_from_dict_w_opt
from ..celapp import app


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


def train_model(model_dict, datas, datas_val, batch_size=32,
                nb_epoch=10, callbacks=None, custom_objects=None):
    """Train a model given hyperparameters and a serialized model"""

    if callbacks is None:
        callbacks = []
    loss = []
    val_loss = []
    # load model
    model = model_from_dict_w_opt(model_dict, custom_objects=custom_objects)

    # fit the model according to the input/output type
    if model.__class__.__name__ is "Graph":
        for d, dv in zip(datas, datas_val):
            h = model.fit(data=d,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=dv)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']

    elif model.__class__.__name__ is "Sequential":
        # unpack data
        for d, dv in zip(datas, datas_val):
            X, y = d['X'], d['y']
            X_val, y_val = dv['X'], dv['y']
            h = model.fit(x=X,
                          y=y,
                          batch_size=batch_size,
                          nb_epoch=nb_epoch,
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=(X_val, y_val))
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']

    return loss, val_loss, model


@app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
def fit(model, data, params):
    """Dummy fit for now"""
    return model


def build(model):
    """Dummy build for now"""
    return model


@app.task
def predict(model, data):
    """Dummy predict for now"""
    return model
