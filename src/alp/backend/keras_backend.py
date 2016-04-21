"""Adaptor for the Keras backend"""

import keras.backend as K

from ..celapp import app
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


def fit(model, data, data_val, *args, **kwargs):
    """Fit a model given hyperparameters and a serialized model"""
    custom_objects = kwargs.pop('custom_objects')
    loss = []
    val_loss = []
    # load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)

    # fit the model according to the input/output type
    if model.__class__.__name__ is "Graph":
        for d, dv in zip(data, data_val):
            h = model.fit(data=d,
                          verbose=1,
                          validation_data=dv,
                          *args,
                          **kwargs)
            loss += h.history['loss']
            if 'val_loss' in h.history:
                val_loss += h.history['val_loss']

    elif model.__class__.__name__ is "Sequential":
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
            max_iter = h.history
    else:
        raise NotImplementedError("This type of mode lis not supported")

    return loss, val_loss, max_iter, model


@app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
def fit2(model, data, data_val, *args, **kwargs):
    """A function to train models given a datagenerator,a serialized model,

    Args:
        model_str(str): the model dumped with the `to_json` method
        data_gen(generator): a generator yielding (mini) batches of train and
            validation data
        offset(int): how many datapoints to burn

    Returns:
        the unique id of the model"""

    from databasesetup import get_models
    from datetime import datetime
    import hashlib
    import json
    import numpy as np

    batch_size = kwargs.pop("batch_size")
    if batch_size is None:
        batch_size = 32
    # convert string to json
    model_str = json.dumps(model)

    # get the models collection
    models = get_models()

    first = data.keys()[0]
    un_data_m = data[first].mean()
    un_data_f = data[first][0]

    # create the model hash from the stringified json
    mh = hashlib.md5()
    mh.update(model_str + str(batch_size))
    hexdi_m = mh.hexdigest()

    # create the data hash
    dh = hashlib.md5()
    dh.update(str(un_data_m) + str(un_data_f))
    hexdi_d = dh.hexdigest()

    params_dump = "/parameters_h5/" + hexdi_m + hexdi_d + '.h5'

    # update the full json
    full_json = {'keras_model': model,
                 'datetime': datetime.now(),
                 'hashed_mod': hexdi_m,
                 'data_id': hexdi_d,
                 'params_dump': params_dump,
                 'batch_size': batch_size,
                 'trained': 0,
                 'data_path': "sent",
                 'root': "sent",
                 'data_s': "sent"}
    mod_id = models.insert_one(full_json).inserted_id

    try:
        loss, val_loss, iters, model = fit(model, data,
                                           data_val,
                                           batch_size=batch_size,
                                           *args, **kwargs)
        upres = models.update({"_id": mod_id}, {'$set': {
            'train_loss': loss,
            'min_tloss': np.min(loss),
            'valid_loss': val_loss,
            'min_vloss': np.min(val_loss),
            'iter_stopped': nb_epoch * len(data),
            'trained': 1,
            'date_finished_trained': datetime.now()
        }})

        model.save_weights(params_dump, overwrite=True)

    except MemoryError as e:
        models.delete_one({'hashed_mod': hexdi_m})
        raise self.retry(exc=exc) # pragma: no cover

    except Exception as e:
        models.update({"_id": mod_id}, {'$set': {'error': 1}})
        raise e
    return hexdi_m, hexdi_d


@app.task
def predict(model, data, *args, **kwargs):
    """Dummy predict for now"""
    return model
