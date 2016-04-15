"""Adaptor for the Keras backend"""

from keras.models import model_from_json
import keras.backend as K
import json
from keras import optimizers


def build_from_json(model_str, custom_objects=None):
    if custom_objects is None:
        custom_objects = []
    model = model_from_json(model_str, custom_objects=custom_objects)
    config = json.loads(model_str)

    if 'optimizer' in config:
        model_name = config.get('class_name')
        # if it has an optimizer, the model is assumed to be compiled
        loss = config.get('loss')

        # if a custom loss function is passed replace it in loss
        if model_name is "Graph":
            for l in loss:
                for c in custom_objects:
                    if loss[l] == c:
                        loss[l] = custom_objects[c]
        elif model_name is "Sequential" and loss in custom_objects:
            loss = custom_objects[loss]

        class_mode = config.get('class_mode')

        optimizer_params = dict([(k, v)
                                 for k, v in config.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name is "Sequential":
            sample_weight_mode = config.get('sample_weight_mode')
            model.compile(loss=loss,
                          optimizer=optimizer,
                          class_mode=class_mode,
                          sample_weight_mode=sample_weight_mode)
        elif model_name is "Graph":
            sample_weight_modes = config.get('sample_weight_modes', {})
            loss_weights = config.get('loss_weights', {})
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_modes=sample_weight_modes,
                          loss_weights=loss_weights)
    return model


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


def train_model(model_str, custom_objects, datas, datas_val, batch_size,
                nb_epoch, callbacks):
    """Train a model given hyperparameters and a serialized model"""

    loss = []
    val_loss = []
    # load model
    model = build_from_json(model_str, custom_objects=custom_objects)

    # fit the model according to the input/output type
    if model.class_name is "Graph":
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

    elif model.class_name is "Sequential":
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
    else:
        raise NotImplementedError('This type of model is not supported')

    return loss, val_loss, model
