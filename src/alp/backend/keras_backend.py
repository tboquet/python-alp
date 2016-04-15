"""Adaptor for the Keras backend"""

from keras.models import model_from_json
import keras.backend as K
import json
from keras import optimizers
import copy
import six


def to_json_w_opt(model):
    config = dict()
    config['config'] = model.to_json()
    if hasattr(model, 'optimizer'):
        config['optimizer'] = model.optimizer.get_config()
    if hasattr(model, 'loss'):
        if type(model.loss) == dict:
            config['loss'] = dict([(k, get_function_name(v))
                                   for k, v in model.loss.items()])
        else:
            config['loss'] = get_function_name(model.loss)

    return config


def build_from_json(model_json, custom_objects=None):
    if custom_objects is None:
        custom_objects = []
    model = model_from_json(model_json['config'],
                            custom_objects=custom_objects)
    config = json.loads(model_json['config'])
    if 'optimizer' in model_json:
        model_name = config.get('class_name')
        print(model_name, model_name is "Sequential")
        # if it has an optimizer, the model is assumed to be compiled
        loss = model_json.get('loss')

        # if a custom loss function is passed replace it in loss
        if model_name == "Graph":
            for l in loss:
                for c in custom_objects:
                    if loss[l] == c:
                        loss[l] = custom_objects[c]
        elif model_name == "Sequential" and loss in custom_objects:
            loss = custom_objects[loss]

        optimizer_params = dict([(
            k, v) for k, v in model_json.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name == "Sequential":
            print("yeah")
            sample_weight_mode = config.get('sample_weight_mode')
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_mode=sample_weight_mode)
        elif model_name == "Graph":
            sample_weight_modes = config.get('sample_weight_modes', {})
            loss_weights = config.get('loss_weights', {})
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_modes=sample_weight_modes,
                          loss_weights=loss_weights)
    return model


def get_function_name(o):
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__


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


def train_model(model_json, custom_objects, datas, datas_val, batch_size,
                nb_epoch, callbacks):
    """Train a model given hyperparameters and a serialized model"""

    loss = []
    val_loss = []
    # load model
    model = build_from_json(model_json, custom_objects=custom_objects)

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
