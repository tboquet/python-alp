"""Adaptor for the Keras backend"""

import keras.backend as K
import six
from keras import optimizers
from keras.utils.layer_utils import layer_from_config

from ..celapp import app
from ..config import PATH_H5


# Serialization utilities

def to_dict_w_opt(model, metrics=None):
    """Serialize a model and add the config of the optimizer and the loss.
    """
    config = dict()
    config_m = model.get_config()
    config['config'] = {
        'class_name': model.__class__.__name__,
        'config': config_m,
    }
    if hasattr(model, 'optimizer'):
        config['optimizer'] = model.optimizer.get_config()
    if hasattr(model, 'loss'):
        name_out = [l.name for l in model.output_layers]
        if isinstance(model.loss, dict):
            print("dict", model.loss)
            config['loss'] = dict([(k, get_function_name(v))
                                   for k, v in model.loss.items()])
        elif isinstance(model.loss, list):
            print("list", model.loss)
            config['loss'] = dict(zip(name_out, [get_function_name(l)
                                                 for l in model.loss]))
        elif hasattr(model.loss, '__call__'):
            print("func", model.loss)
            config['loss'] = dict(zip(name_out,
                                      [get_function_name(model.loss)]))
        elif isinstance(model.loss, six.string_types):
            print("string", model.loss)
            config['loss'] = dict(zip(name_out,
                                      [get_function_name(model.loss)]))
    if metrics is not None:
        config['metrics'] = metrics

    return config


def model_from_dict_w_opt(model_dict, custom_objects=None):
    """Builds a model from a serialized model using ``to_dict_w_opt`
    """
    if custom_objects is None:
        custom_objects = {}

    model = layer_from_config(model_dict['config'],
                              custom_objects=custom_objects)

    if 'optimizer' in model_dict:
        metrics = model_dict.get("metrics")
        model_name = model_dict['config'].get('class_name')
        # if it has an optimizer, the model is assumed to be compiled
        loss = model_dict.get('loss')

        # if a custom loss function is passed replace it in loss
        for l in loss:
            for c in custom_objects:
                if loss[l] == c:
                    loss[l] = custom_objects[c]

        optimizer_params = dict([(
            k, v) for k, v in model_dict.get('optimizer').items()])
        optimizer_name = optimizer_params.pop('name')
        optimizer = optimizers.get(optimizer_name, optimizer_params)

        if model_name == "Sequential":
            sample_weight_mode = model_dict.get('sample_weight_mode')
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_mode=sample_weight_mode,
                          metrics=metrics)
        elif model_name == "Graph":
            sample_weight_modes = model_dict.get('sample_weight_modes', None)
            loss_weights = model_dict.get('loss_weights', None)
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_modes=sample_weight_modes,
                          loss_weights=loss_weights)
        elif model_name == "Model":
            sample_weight_mode = model_dict.get('sample_weight_mode')
            loss_weights = model_dict.get('loss_weights', None)
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_mode=sample_weight_mode,
                          loss_weights=loss_weights,
                          metrics=metrics)
    return model


def get_function_name(o):
    """Utility function to return the model's name
    """
    if isinstance(o, six.string_types):
        return o
    else:
        return o.__name__


# core utilities

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


def train(model, data, data_val, *args, **kwargs):
    """Fit a model given hyperparameters and a serialized model"""
    custom_objects = None

    if 'custom_objects' in kwargs:
        custom_objects = kwargs.pop('custom_objects')
    loss = []
    val_loss = []
    # load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)
    mod_name = model.__class__.__name__

    # fit the model according to the input/output type
    if mod_name is "Graph":
        for d, dv in zip(data, data_val):
            h = model.fit(data=d,
                          verbose=1,
                          validation_data=dv,
                          *args,
                          **kwargs)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']
        max_iter = h.epoch[-1]

    elif mod_name is "Sequential":
        for d, dv in zip(data, data_val):
            X, y = d['X'], d['y']
            X_val, y_val = dv['X'], dv['y']
            h = model.fit(x=X,
                          y=y,
                          verbose=1,
                          validation_data=(X_val, y_val),
                          *args,
                          **kwargs)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']
        max_iter = h.epoch[-1]
    elif mod_name is "Model":
        for d, dv in zip(data, data_val):
            X, y = d['X'], d['y']
            X_val, y_val = dv['X'], dv['y']
            h = model.fit(x=X,
                          y=y,
                          verbose=1,
                          validation_data=(X_val, y_val),
                          *args,
                          **kwargs)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']
        max_iter = h.epoch[-1]
    else:
        raise NotImplementedError("This type of model"
                                  "is not supported: {}".format(mod_name))

    return loss, val_loss, max_iter, model


@app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
def fit(model, data, data_val, *args, **kwargs):
    """A function to train models given a datagenerator,a serialized model,

    Args:
        model_str(str): the model dumped with the `to_json` method
        data_gen(generator): a generator yielding (mini) batches of train and
            validation data
        offset(int): how many datapoints to burn

    Returns:
        the unique id of the model"""

    from ..databasecon import get_models
    from datetime import datetime
    import hashlib
    import json
    import numpy as np

    if kwargs.get("batch_size") is None:
        kwargs['batch_size'] = 32

    # convert dict to json string
    print(model)
    model_str = json.dumps(model)

    # get the models collection
    models = get_models()

    first = list(data[0].keys())[0]
    un_data_m = data[0][first].mean()
    un_data_f = data[0][first][0]

    # create the model hash from the stringified json
    mh = hashlib.md5()
    str_concat_m = str(model_str) + str(kwargs['batch_size'])
    mh.update(str_concat_m.encode('utf-8'))
    hexdi_m = mh.hexdigest()

    # create the data hash
    dh = hashlib.md5()
    str_concat_d = str(un_data_m) + str(un_data_f)
    dh.update(str_concat_d.encode('utf-8'))
    hexdi_d = dh.hexdigest()

    params_dump = PATH_H5 + hexdi_m + hexdi_d + '.h5'

    # update the full json
    full_json = {'keras_model': model,
                 'datetime': datetime.now(),
                 'hashed_mod': hexdi_m,
                 'data_id': hexdi_d,
                 'params_dump': params_dump,
                 'batch_size': kwargs['batch_size'],
                 'trained': 0,
                 'data_path': "sent",
                 'root': "sent",
                 'data_s': "sent"}
    mod_id = models.insert_one(full_json).inserted_id

    try:
        loss, val_loss, iters, model = train(model, data,
                                             data_val,
                                             *args, **kwargs)
        models.update({"_id": mod_id}, {'$set': {
            'train_loss': loss,
            'min_tloss': np.min(loss),
            'valid_loss': val_loss,
            'min_vloss': np.min(val_loss),
            'iter_stopped': iters * len(data),
            'trained': 1,
            'date_finished_trained': datetime.now()
        }})

        model.save_weights(params_dump, overwrite=True)

    except Exception as e:
        models.update({"_id": mod_id}, {'$set': {'error': 1}})
        raise e
    return hexdi_m, hexdi_d


@app.task
def predict(model, data, *args, **kwargs):
    """Dummy predict for now"""
    return model
