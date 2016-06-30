"""
Adaptor for the Keras backend
=============================

Serialization
~~~~~~~~~~~~~

The models are serialized in a json format and pushed in a database.
RabbitMQ also receives a training message and will release it to an
available worker.

.. note:

    The syntax will evolve a lot and changes have to be expected.

Compilation & cache
~~~~~~~~~~~~~~~~~~~

The models are compiled on the fly after the build. If the model is already
compiled and in the `COMPILED_MODEL` dictionnary mapping the models id to the
in memory compiled function, this function is used instead.

----------------------------------------------------------------------------
"""

import types

import dill
import marshal
import six
from six.moves import zip as szip

from ..appcom import _path_h5
from ..backend import common as cm
from ..celapp import app

COMPILED_MODELS = dict()
TO_SERIALIZE = ['custom_objects']


# general utilities

def get_backend():
    import keras as CB
    return CB


def serialize(cust_obj):
    """A function to serialize custom objects passed to a model

    Args:
        cust_obj(callable): a custom layer or function to serialize

    Returns:
        a dict of the serialized components of the object"""
    ser_func = dict()
    if isinstance(cust_obj, types.FunctionType):
        func_code = six.get_function_code(cust_obj)
        func_code_d = marshal.dumps(func_code).decode('raw_unicode_escape')
        ser_func['func_code_d'] = func_code_d
        ser_func['name_d'] = marshal.dumps(cust_obj.__name__)
        ser_func['args_d'] = marshal.dumps(six.get_function_defaults(cust_obj))
        ser_func['clos_d'] = dill.dumps(six.get_function_closure(cust_obj))
        ser_func['type_obj'] = 'func'
    else:
        func_code_d = dill.dumps(cust_obj).decode('raw_unicode_escape')
        ser_func['func_code_d'] = func_code_d
        ser_func['name_d'] = None
        ser_func['args_d'] = None
        ser_func['clos_d'] = None
        ser_func['type_obj'] = 'class'
    return ser_func


def deserialize(name_d, func_code_d, args_d, clos_d, type_obj):
    """A function to deserialize an object serialized with the serialize
    function.

    Args:
        name_d(unicode): the dumped name of the object
        func_code_d(unicode): the dumped byte code of the function
        args_d(unicode): the dumped information about the arguments
        clos_d(unicode): the dumped information about the function closure

    Returns:
        a deserialized object"""
    if type_obj == 'func':
        name = marshal.loads(name_d)
        code = marshal.loads(func_code_d.encode('raw_unicode_escape'))
        args = marshal.loads(args_d)
        clos = dill.loads(clos_d)
        loaded_obj = types.FunctionType(code, globals(), name, args, clos)
    else:
        loaded_obj = dill.loads(func_code_d.encode('raw_unicode_escape'))

    return loaded_obj


# Serialization utilities

def to_dict_w_opt(model, metrics=None):
    """Serialize a model and add the config of the optimizer and the loss.

    Args:
        model(keras.Model): the model to serialize
        metrics(list, optionnal): a list of metrics to monitor

    Returns:
        a dictionnary of the serialized model
    """
    config = dict()
    config_m = model.get_config()

    config['config'] = {
        'class_name': model.__class__.__name__,
        'config': config_m,
    }
    if hasattr(model, 'optimizer'):
        config['optimizer'] = model.optimizer.get_config()
        config['metrics'] = []
        config['ser_metrics'] = dict()
    if hasattr(model, 'loss'):
        name_out = [l.name for l in model.output_layers]
        if isinstance(model.loss, dict):
            config['loss'] = dict([(k, get_function_name(v))
                                   for k, v in model.loss.items()])
        elif isinstance(model.loss, list):
            config['loss'] = dict(zip(name_out, [get_function_name(l)
                                                 for l in model.loss]))
        elif hasattr(model.loss, '__call__'):
            config['loss'] = dict(zip(name_out,
                                      [get_function_name(model.loss)]))
        elif isinstance(model.loss, six.string_types):
            config['loss'] = dict(zip(name_out,
                                      [get_function_name(model.loss)]))
    if metrics is not None:
        for m in metrics:
            if isinstance(m, six.string_types):
                config['metrics'].append(m)
            else:
                config['ser_metrics'][m.__name__] = serialize(m)
    return config


def model_from_dict_w_opt(model_dict, custom_objects=None):
    """Builds a model from a serialized model using `to_dict_w_opt`

    Args:
        model_dict(dict): a serialized Keras model
        custom_objects(dict, optionnal): a dictionnary mapping custom objects
            names to custom objects (Layers, functions, etc.)

    Returns:
        A Keras.Model which is compiled if the information about the optimizer
        is available.

    """
    from keras import optimizers
    from keras.utils.layer_utils import layer_from_config

    if custom_objects is None:
        custom_objects = dict()

    custom_objects = {k: deserialize(**custom_objects[k])
                      for k in custom_objects}

    model = layer_from_config(model_dict['config'],
                              custom_objects=custom_objects)

    if 'optimizer' in model_dict:
        metrics = model_dict.get("metrics", [])
        ser_metrics = model_dict.get("ser_metrics", dict())
        metrics += [deserialize(**m) for k, m in ser_metrics.items()]
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

    Args:
        o(object): an object to check

    Returns:
        The name(str) of the object
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
        mod(keras.models): a Model, Sequential, or Graph (deprecated) model

    Returns:
        a Keras (Theano or Tensorflow) function
    """
    import keras.backend as K
    if mod.uses_learning_phase:
        tensors = mod.inputs + [K.learning_phase()]
    else:
        tensors = mod.inputs
    return K.function(tensors, mod.outputs, updates=mod.state_updates)


def train(model, data, data_val, generator=False, *args, **kwargs):
    """Fit a model given hyperparameters and a serialized model

    Args:
        model(dict): a serialized keras.Model
        data(list): a list of dict mapping inputs and outputs to lists or
            dictionnaries mapping the inputs names to np.arrays
        data_val(list): same structure than `data` but for validation

    Returns:
        the loss (list), the validation loss (list), the number of iterations,
        and the model
        """
    results = dict()
    results['metrics'] = dict()
    custom_objects = None
    fit_gen_val = False
    suf = 'val_'

    if 'custom_objects' in kwargs:
        custom_objects = kwargs.pop('custom_objects')

    # load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)

    metrics_names = model.metrics_names
    for metric in metrics_names:
        results['metrics'][metric] = []
        results['metrics']['val_' + metric] = []
    mod_name = model.__class__.__name__

    if generator:
        data = [cm.transform_gen(dv, mod_name) for dv in data]
        kwargs.pop('batch_size')

    val_gen = (hasattr(data_val[-1], 'next') or
               hasattr(data_val[-1], '__next__'))

    val_gen += 'fuel' in repr(data_val[-1])

    if val_gen:
        if generator:
            data_val = [cm.transform_gen(dv, mod_name) for dv in data_val]
            fit_gen_val = True
        else:
            raise Exception("You should also pass a generator for the training"
                            " data.")

    # fit the model according to the input/output type
    if mod_name is "Graph":
        for d, dv in szip(data, data_val):
            if generator:
                h = model.fit_generator(generator=d,
                                        validation_data=dv,
                                        *args,
                                        **kwargs)
            else:
                h = model.fit(data=d,
                              validation_data=dv,
                              *args,
                              **kwargs)
            for metric in metrics_names:
                results['metrics'][metric] += h.history[metric]
                results['metrics']['val_' + metric] += h.history[metric]
        results['metrics']['iter'] = h.epoch[-1] * len(data)

    elif mod_name is "Sequential" or mod_name is "Model":
        for d, dv in szip(data, data_val):
            if not fit_gen_val:
                dv = (dv['X'], dv['y'])
            if generator:
                h = model.fit_generator(generator=d,
                                        validation_data=dv,
                                        *args,
                                        **kwargs)
            else:
                X, y = d['X'], d['y']
                h = model.fit(x=X,
                              y=y,
                              validation_data=dv,
                              *args,
                              **kwargs)
            for metric in metrics_names:
                results['metrics'][metric] += h.history[metric]
                results['metrics'][suf + metric] += h.history[suf + metric]
        results['metrics']['iter'] = h.epoch[-1] * len(data)
    else:
        raise NotImplementedError("This type of model"
                                  "is not supported: {}".format(mod_name))
    return results, model


@app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
def fit(backend_name, backend_version, model, data, data_hash, data_val,
        generator=False, *args, **kwargs):
    """a function to train models given a datagenerator,a serialized model,

    args:
        backend_name(str): the model dumped with the `to_json` method
        backend_version(str): the backend version
        model(keras.model): a keras model
        data(list): a list of np.arrays for training
        data_val(list): a list of np.arrays for validation

    returns:
        results similar to what the fit method of keras would return"""

    from alp import dbbackend as db
    from datetime import datetime
    import alp.backend.common as cm
    import numpy as np
    if kwargs.get("batch_size") is None:
        kwargs['batch_size'] = 32

    batch_size = kwargs['batch_size']

    model_c = cm.clean_model(model)
    hexdi_m = cm.create_model_hash(model_c, batch_size)
    params_dump = cm.create_param_dump(_path_h5, hexdi_m, data_hash)

    # update the full json
    full_json = {'backend_name': backend_name,
                 'backend_version': backend_version,
                 'model_arch': model_c['model_arch'],
                 'datetime': datetime.now(),
                 'mod_id': hexdi_m,
                 'data_id': data_hash,
                 'params_dump': params_dump,
                 'batch_size': kwargs['batch_size'],
                 'trained': 0,
                 'data_path': "sent",
                 'root': "sent",
                 'data_s': "sent"}
    mod_id = db.insert(full_json)

    try:
        results, model = train(model['model_arch'], data,
                               data_val,
                               generator=generator,
                               *args, **kwargs)
        res_dict = {
            'iter_stopped': results['metrics']['iter'],
            'trained': 1,
            'date_finished_training': datetime.now()}
        for metric in results['metrics']:
            res_dict[metric] = results['metrics'][metric]
            if metric in ['loss', 'val_loss']:
                res_dict[metric] = np.min(results['metrics'][metric])
        db.update({"_id": mod_id}, {'$set': res_dict})

        model.save_weights(params_dump, overwrite=True)
        results['model_id'] = hexdi_m
        results['data_id'] = data_hash
        results['params_dump'] = params_dump

    except Exception:
        db.update({"_id": mod_id}, {'$set': {'error': 1}})
        raise
    return results


@app.task
def predict(model, data, *args, **kwargs):
    """Make predictions given a model and data

    Args:
        model(dict): a serialized keras models
        data(list, dict, np.array): data to be passed as a dictionary mapping
            inputs names to np.arrays or a list of arrays or an arrays

    Returns:
        an np.array of predictions
    """

    custom_objects = kwargs.get('custom_objects')

    model_name = model['model_arch']['config'].get('class_name')
    # check if the predict function is already compiled
    if model['mod_id'] in COMPILED_MODELS:
        pred_function = COMPILED_MODELS[model['mod_id']]['pred']
        model_k = COMPILED_MODELS[model['mod_id']]['model']
        learning_phase = COMPILED_MODELS[model['mod_id']]['learning_phase']
    else:
        # get the model arch
        model_dict = model['model_arch']

        model_dict.pop('optimizer')

        # load model
        model_k = model_from_dict_w_opt(model_dict,
                                        custom_objects=custom_objects)
        # load the weights
        model_k.load_weights(model['params_dump'])

        # build the prediction function
        pred_function = build_predict_func(model_k)
        COMPILED_MODELS[model['mod_id']] = dict()
        COMPILED_MODELS[model['mod_id']]['pred'] = pred_function
        COMPILED_MODELS[model['mod_id']]['model'] = model_k
        learning_phase = model_k.uses_learning_phase
        COMPILED_MODELS[model['mod_id']]['learning_phase'] = learning_phase

    # predict according to the input/output type
    if model_name == 'Graph':
        if isinstance(data, dict):
            data = [data[n] for n in model_k.input_names]
        if not isinstance(data, list):
            data = [data]
    elif model_name == 'Sequential':
        if not isinstance(data, list):
            data = [data]
    elif model_name == 'Model':
        if isinstance(data, dict):
            data = [data[k] for k in model_k.input_names]
        elif not isinstance(data, list):
            data = [data]
    else:
        raise NotImplementedError(
            '{}: This type of model is not supported'.format(model_name))
    if learning_phase:
        data.append(0.)
    return pred_function(data)
