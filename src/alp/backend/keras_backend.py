"""
Adaptor for the Keras backend
=============================

Compilation & cache
~~~~~~~~~~~~~~~~~~~

The models are compiled on the fly after the build. If the model is already
compiled and in the `COMPILED_MODEL` dictionnary mapping the models id to the
in memory compiled function, this function is used instead.

----------------------------------------------------------------------------
"""

import inspect
import types

import dill
import numpy as np
import six

from six.moves import zip as szip

from ..appcom import _path_h5
from ..appcom.utils import check_gen
from ..backend import common as cm
from ..celapp import RESULT_SERIALIZER
from ..celapp import app

try:  # pragma: no cover
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


COMPILED_MODELS = dict()
TO_SERIALIZE = ['custom_objects', 'callbacks']


# general utilities

def get_backend():
    import keras as CB
    return CB


def check_validation(dv):
    validation = True
    if dv is None:
        validation = False
    return(validation)


def save_params(model, filepath):
    model.save_weights(filepath, overwrite=True)


def serialize(cust_obj):
    """A function to serialize custom objects passed to a model

    Args:
        cust_obj(callable): a custom layer or function to serialize

    Returns:
        a dict of the serialized components of the object"""
    ser_func = dict()
    if isinstance(cust_obj, types.FunctionType):

        func_code = six.get_function_code(cust_obj)
        func_code_d = dill.dumps(func_code).decode('raw_unicode_escape')
        ser_func['func_code_d'] = func_code_d
        ser_func['name_d'] = pickle.dumps(
            cust_obj.__name__).decode('raw_unicode_escape')
        ser_func['args_d'] = pickle.dumps(
            six.get_function_defaults(cust_obj)).decode('raw_unicode_escape')
        clos = dill.dumps(
            six.get_function_closure(cust_obj)).decode('raw_unicode_escape')
        ser_func['clos_d'] = clos
        ser_func['type_obj'] = 'func'
    else:
        if hasattr(cust_obj, '__module__'):  # pragma: no cover
            cust_obj.__module__ = '__main__'
        ser_func['name_d'] = None
        ser_func['args_d'] = None
        ser_func['clos_d'] = None
        ser_func['type_obj'] = 'class'
        loaded = dill.dumps(cust_obj).decode('raw_unicode_escape')
        ser_func['func_code_d'] = loaded
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
        name = pickle.loads(name_d.encode('raw_unicode_escape'))
        code = dill.loads(func_code_d.encode('raw_unicode_escape'))
        args = pickle.loads(args_d.encode('raw_unicode_escape'))
        clos = dill.loads(clos_d.encode('raw_unicode_escape'))
        loaded_obj = types.FunctionType(code, globals(), name, args, clos)
    else:  # pragma: no cover
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
        config['optimizer']['name'] = model.optimizer.__class__.__name__
        config['metrics'] = []
        config['ser_metrics'] = []
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
        else:  # pragma: no cover
            raise TypeError('Loss must be a list a string or a callable.')

    if metrics is not None:
        for m in metrics:
            if isinstance(m, six.string_types):
                config['metrics'].append(m)
            else:
                config['ser_metrics'].append(m.__name__)
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

    for k in custom_objects:
        if inspect.isfunction(custom_objects[k]):
            custom_objects[k] = custom_objects[k]()

    model = layer_from_config(model_dict['config'],
                              custom_objects=custom_objects)

    if 'optimizer' in model_dict:
        metrics = model_dict.get("metrics", [])
        ser_metrics = model_dict.get("ser_metrics", [])
        for k in custom_objects:
            if inspect.isfunction(custom_objects[k]):
                function_name = custom_objects[k].__name__
                if k in ser_metrics or function_name in ser_metrics:
                    metrics.append(custom_objects[k])
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
        elif model_name == "Model":
            sample_weight_mode = model_dict.get('sample_weight_mode')
            loss_weights = model_dict.get('loss_weights', None)
            model.compile(loss=loss,
                          optimizer=optimizer,
                          sample_weight_mode=sample_weight_mode,
                          loss_weights=loss_weights,
                          metrics=metrics)
        else:  # pragma: no cover
            raise Exception('{} model, must be in Sequential, '
                            'Model'.format(model_name))

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
        mod(keras.models): a Model or Sequential model

    Returns:
        a Keras (Theano or Tensorflow) function
    """
    import keras.backend as K
    if mod.uses_learning_phase:
        tensors = mod.inputs + [K.learning_phase()]
    else:
        tensors = mod.inputs
    return K.function(tensors, mod.outputs, updates=mod.state_updates)


def train(model, data, data_val, size_gen, generator=False, *args, **kwargs):
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
    if generator:
        from six.moves import reload_module as sreload
        import theano
        sreload(theano)

    results = dict()
    results['metrics'] = dict()
    custom_objects = None
    fit_gen_val = False
    suf = 'val_'

    if 'custom_objects' in kwargs:
        custom_objects = kwargs.pop('custom_objects')

    # load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)

    if 'callbacks' in kwargs:
        callbacks = kwargs.pop('callbacks')

    if callbacks is None:
        callbacks = []

    callbacks = [deserialize(**callback)
                 for callback in callbacks]

    for i, callback in enumerate(callbacks):
        if inspect.isfunction(callback):
            callbacks[i] = callback()

    metrics_names = model.metrics_names
    for metric in metrics_names:
        results['metrics'][metric] = []
        results['metrics'][suf + metric] = []
    mod_name = model.__class__.__name__

    if generator:
        data = [pickle.loads(d.encode('raw_unicode_escape')) for d in data]
        data = [cm.transform_gen(d, mod_name) for d in data]
        kwargs.pop('batch_size')

    if all(v is None for v in data_val):
        val_gen = 0
    else:
        val_gen = check_gen(data_val)

    if val_gen > 0:
        if generator:
            data_val = [pickle.loads(dv.encode('raw_unicode_escape'))
                        for dv in data_val]
            data_val = [cm.transform_gen(dv, mod_name) for dv in data_val]
            for i, check in enumerate(size_gen):
                if check is 1:
                    data_val[i] = next(data_val[i])
            fit_gen_val = True
        else:
            raise Exception("You should also pass a generator for the training"
                            " data.")

    # fit the model according to the input/output type

    if mod_name is "Sequential" or mod_name is "Model":
        for d, dv in szip(data, data_val):
            validation = check_validation(dv)
            if not fit_gen_val:
                if dv is not None:
                    dv = (dv['X'], dv['y'])
            if generator:
                h = model.fit_generator(generator=d,
                                        validation_data=dv,
                                        callbacks=callbacks,
                                        *args,
                                        **kwargs)
            else:
                X, y = d['X'], d['y']
                h = model.fit(x=X,
                              y=y,
                              validation_data=dv,
                              callbacks=callbacks,
                              *args,
                              **kwargs)
            for metric in metrics_names:
                results['metrics'][metric] += h.history[metric]
                if validation:
                    results['metrics'][
                        suf + metric] += h.history[suf + metric]
                else:
                    results['metrics'][suf + metric] += [np.nan] * \
                        len(h.history[metric])
        results['metrics']['iter'] = h.epoch[-1] * len(data)
    else:
        raise NotImplementedError("This type of model"
                                  "is not supported: {}".format(mod_name))
    return results, model


@app.task(bind=True, default_retry_delay=60 * 10, max_retries=3,
          rate_limit='20/s', queue='keras')
def fit(self, backend_name, backend_version, model, data, data_hash, data_val,
        size_gen, generator=False, *args, **kwargs):
    """A function to train models given a datagenerator,a serialized model,

    Args:
        backend_name(str): the model dumped with the `to_json` method
        backend_version(str): the backend version
        model(keras.model): a keras model
        data(list): a list of np.arrays for training
        data_val(list): a list of np.arrays for validation

    Returns:
        results similar to what the fit method of keras would return"""
    from alp import dbbackend as db
    from datetime import datetime
    import alp.backend.common as cm
    import keras.backend as K
    if K.backend() == 'tensorflow' and cm.on_worker():  # pragma: no cover
        import tensorflow as tf
        K.clear_session()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)

    if kwargs.get("batch_size") is None:
        kwargs['batch_size'] = 32

    if kwargs.get("overwrite") is None:  # pragma: no cover
        overwrite = False
    else:
        overwrite = kwargs.pop("overwrite")

    batch_size = kwargs['batch_size']

    model_c = cm.clean_model(model)

    hexdi_m, params_dump = cm.make_all_hash(model_c, batch_size, data_hash,
                                            _path_h5)

    # update the full json
    full_json_model = {'backend_name': backend_name,
                       'backend_version': backend_version,
                       'model_arch': model_c['model_arch'],
                       'datetime': datetime.now(),
                       'mod_id': hexdi_m,
                       'data_id': data_hash,
                       'params_dump': params_dump,
                       'batch_size': kwargs['batch_size'],
                       'trained': 0,
                       'mod_data_id': hexdi_m + data_hash,
                       'task_id': self.request.id}

    mod_id = db.insert(full_json_model, db.get_models(), upsert=overwrite)

    if generator is True:
        full_json_data = {'mod_data_id': hexdi_m + data_hash,
                          'data_id': data_hash,
                          'data': data}
        db.insert(full_json_data, db.get_generators(), upsert=overwrite)

    try:
        results, res_dict = cm.train_pipe(train, save_params, model, data,
                                          data_val, generator, size_gen,
                                          params_dump, data_hash, hexdi_m,
                                          *args, **kwargs)

        db.update({'_id': mod_id}, {'$set': res_dict})

    except Exception:
        db.update({'_id': mod_id}, {'$set': {'error': 1}})
        raise
    return results


@app.task(queue='keras')
def predict(model, data, async, *args, **kwargs):
    """Make predictions given a model and data

    Args:
        model(dict): a serialized keras models
        data(list, dict, np.array): data to be passed as a dictionary mapping
            inputs names to np.arrays or a list of arrays or an arrays

    Returns:
        an np.array of predictions
    """
    import alp.backend.common as cm
    import keras.backend as K

    from keras.engine.training import make_batches

    if K.backend() == 'tensorflow' and cm.on_worker():  # pragma: no cover
        import tensorflow as tf
        K.clear_session()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)

    json_serializer = RESULT_SERIALIZER == 'json'
    if kwargs.get("batch_size") is None:  # pragma: no cover
        kwargs['batch_size'] = 32

    batch_size = kwargs['batch_size']

    custom_objects = kwargs.get('custom_objects')

    model_name = model['model_arch']['config'].get('class_name')
    # check if the predict function is already compiled
    m_id = model['mod_id'] + model['data_id']

    if m_id in COMPILED_MODELS:
        pred_function = COMPILED_MODELS[m_id]['pred']
        model_k = COMPILED_MODELS[m_id]['model']
        learning_phase = COMPILED_MODELS[m_id]['learning_phase']
        output_shape = COMPILED_MODELS[m_id]['model'].output_shape
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
        COMPILED_MODELS[m_id] = dict()
        COMPILED_MODELS[m_id]['pred'] = pred_function
        COMPILED_MODELS[m_id]['model'] = model_k
        learning_phase = model_k.uses_learning_phase
        COMPILED_MODELS[m_id]['learning_phase'] = learning_phase
        output_shape = model_k.output_shape

    # predict according to the input/output type
    if model_name == 'Sequential':
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

    # Predict by batch to control GPU memory
    len_data = len(data[0])
    batches = make_batches(len_data, batch_size)
    index_array = np.arange(len_data)
    results_array = np.empty((len_data, ) + output_shape[1:])
    for batch_start, batch_end in batches:
        batch_ids = index_array[batch_start:batch_end]
        data_b = [d[batch_ids] for d in data]
        if learning_phase:
            data_b.append(0.)
        batch_prediction = pred_function(data_b)
        if isinstance(batch_prediction, list):  # pragma: no cover
            batch_prediction = batch_prediction[0]
        results_array[batch_ids] = batch_prediction
    if async and json_serializer:
        results_array = results_array.tolist()
    return results_array
