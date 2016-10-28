"""
Adaptor for the sklearn backend
=============================
"""

import copy

import h5py
import numpy as np
import pickle
from six.moves import zip as szip
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Lars
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Ridge

from ..appcom import _path_h5
from ..appcom.utils import check_gen
from ..celapp import app
from ..backend import common as cm

SUPPORTED = [LogisticRegression, LinearRegression, Ridge, Lasso,
             Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge,
             ARDRegression, LinearDiscriminantAnalysis,
             QuadraticDiscriminantAnalysis, KernelRidge]

keyval = dict()
for m in SUPPORTED:
    keyval[str(type(m()))[8:][:-2]] = m()


COMPILED_MODELS = dict()
TO_SERIALIZE = ['custom_objects']


# general utilities

def get_backend():
    import sklearn as SK
    return SK


def save_params(model, filepath):
    """ Dumps the attributes of the (generally fitted) model
        in a h5 file.

    Args:
        model(sklearn.BaseEstimator): a sklearn model (in SUPPORTED).
        filepath(string): the file name where the attributes should be written.
    """

    attr = model.__dict__
    dict_params = dict()
    for k, v in attr.items():
        if k[-1:] == '_':
            dict_params[k] = typeconversion(v)

    f = h5py.File(filepath, 'w')
    for k, v in dict_params.items():
        if v is not None:
            f[k] = v
        # so far seen only in Ridge when solver is not sag or lsqr.

    f.flush()
    f.close()


def load_params(model, filepath):
    """ Load the attributes that have been dumped in a h5 file in a model.

    Args:
        model(sklearn.BaseEstimator): a sklearn model (in SUPPORTED).
        filepath(string): the file name where the attributes should be read.
    Returns:
        the model with updated parameters.
    """

    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        with v.astype(v.dtype):
            if v.shape is not ():
                out = v[:]
            else:
                out = v[()]
            setattr(model, k, out)

    f.flush()
    f.close()
    return model


def typeconversion(v):
    """Utility function to ease serialization of custom types
        (namely np.types)

    Args:
        v(np.ndarray, list, other) : the object to return as a jsonable object.
        If the type of v is not a np.ndarray or a list, the type of the
         returned object is unchanged.

    Returns:
        a jsonable object, which type depends on the type of v
    """

    if isinstance(v, np.ndarray):
        return v.tolist()

    elif isinstance(v, list):
        if len(v) == 0:
            return v
        else:
            if isinstance(v[0], np.integer):
                return [int(vv) for vv in v]
            elif isinstance(v[0], np.float):  # pragma: no cover
                return [float(vv) for vv in v]
            elif isinstance(v[0], np.ndarray):
                return [vv.tolist() for vv in v]
            else:  # pragma: no cover
                return v
    else:
        return v


def to_dict_w_opt(model, metrics=None):
    """Serializes a sklearn model. Saves the parameters,
        not the attributes.

    Args:
        model(sklearn.BaseEstimator): the model to serialize,
            must be in SUPPORTED
        metrics(list, optionnal): a list of metrics to monitor

    Returns:
        a dictionnary of the serialized model
    """

    config = dict()
    typestring = str(type(model))[8:][:-2]
    config['config'] = typestring

    attr = model.__dict__

    for k, v in attr.items():
        # check if parameter or attribute
        if k[-1:] == '_':
            # do not store attributes
            pass
        else:
            config[k] = typeconversion(v)

    return config


def model_from_dict_w_opt(model_dict, custom_objects=None):
    """Builds a sklearn model from a serialized model using `to_dict_w_opt`

    Args:
        model_dict(dict): a serialized sklearn model
        custom_objects(dict, optionnal): a dictionnary mapping custom objects
            names to custom objects (callables, etc.)

    Returns:
        A new sklearn.BaseEstimator (in SUPPORTED) instance. The attributes
        are not loaded.

    """
    if custom_objects is None:
        custom_objects = dict()

    # custom_objects = {k: deserialize(k, custom_objects[k])
    #                   for k in custom_objects}

    # safety check
    if model_dict['config'] not in keyval:
        raise NotImplementedError("sklearn model not supported.")

    # create a new instance of the appropriate model type
    model = copy.deepcopy(keyval[model_dict['config']])

    # load the parameters
    for k, v in model_dict.items():
        if isinstance(v, list):  # pragma: no cover
            setattr(model, k, np.array(v))
        else:
            setattr(model, k, v)

    return model


def train(model, data, data_val, generator=False, *args, **kwargs):
    """Fit a model given parameters and a serialized model

    Args:
        model(dict): a serialized sklearn model
        data(list): a list of dict mapping inputs and outputs to lists or
            dictionnaries mapping the inputs names to np.arrays
        data_val(list): same structure than `data` but for validation

    Returns:
        the loss (list), the validation loss (list), the number of iterations,
        and the model
        """

    # Local variables
    from sklearn.metrics import mean_absolute_error

    metrics = []
    results = dict()
    results['metrics'] = dict()
    custom_objects = None
    predondata = []
    predonval = []
    fit_gen_val = False

    metrics.append(mean_absolute_error)
    for metric in metrics:
        results['metrics'][metric.__name__] = []
        results['metrics']['val_' + metric.__name__] = []
    # Load custom_objects and metrics
    if 'custom_objects' in kwargs:  # pragma: no cover
        custom_objects = kwargs.pop('custom_objects')

    if 'metrics' in kwargs:  # pragma: no cover
        print("metrics not supported in sklearn_backend.")
        # metrics = kwargs.pop('metrics')

    # Load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)

    if generator:
        data = [pickle.loads(d) for d in data]
        data = [cm.transform_gen(d, 'Sklearn') for d in data]

    val_gen = check_gen(data_val)

    if val_gen:
        if generator:
            data_val = [pickle.loads(dv) for dv in data_val]
            data_val = [cm.transform_gen(dv, 'Sklearn') for dv in data_val]
            fit_gen_val = True
        else:
            raise Exception("You should also pass a generator for the training"
                            " data.")

    # Fit the model
    for d, dv in szip(data, data_val):

        if fit_gen_val:
            print(dv)
            X_val, y_val = dv
        else:
            X_val, y_val = dv['X'], dv['y']

        if generator:
            X, y = d
        else:
            X, y = d['X'], d['y']

        model.fit(X, y, *args, **kwargs)
        predondata.append(model.predict(d['X']))
        predonval.append(model.predict(dv['X']))

    # Validates the model
    # So far, only the mae is supported.
    for metric in metrics:
        for d, dv, pda, pva in szip(data, data_val, predondata, predonval):
            if fit_gen_val:
                X_val, y_val = dv
            else:
                X_val, y_val = dv['X'], dv['y']

            if generator:
                X, y = d
            else:
                X, y = d['X'], d['y']
            results['metrics'][metric.__name__].append(metric(y, pda))
            results['metrics']['val_' + metric.__name__].append(metric(y_val,
                                                                       pva))

    results['metrics']['iter'] = np.nan

    return results, model


@app.task(bind=True, default_retry_delay=60 * 10, max_retries=3,
          rate_limit='120/m', queue='sklearn')
def fit(self, backend_name, backend_version, model, data, data_hash,
        data_val, generator=False, *args, **kwargs):
    """A function that takes a model and data (with validation),
        then applies the 'train' method if possible.
        The parameters are updated in case of success.

    Args:
        backend_name :
        backend_version :
        model (sklearn.BaseEstimator) : the sklearn model to be trained.
        data(list): a list of dict mapping inputs and outputs to lists or
            dictionnaries mapping the inputs names to np.arrays
        data_val(list): same structure than `data` but for validation

    Returns:
        hexdi_m : the hex hash of the model
        hexdi_d : the hex hash of the data
        params_dump : the name of the file where the attributes are dumped"""

    from alp import dbbackend as db
    import alp.backend.common as cm
    from datetime import datetime

    if kwargs.get("overwrite") is None:  # pragma: no cover
        overwrite = False
    else:
        overwrite = kwargs.pop("overwrite")

    hexdi_m, params_dump = cm.make_all_hash(model, 0, data_hash, _path_h5)

    # update the full json
    full_json = {'backend_name': backend_name,
                 'backend_version': backend_version,
                 'model_arch': model['model_arch'],
                 'datetime': datetime.now(),
                 'mod_id': hexdi_m,
                 'data_id': data_hash,
                 'params_dump': params_dump,
                 'trained': 0,
                 'mod_data_id': hexdi_m + data_hash,
                 'task_id': self.request.id}

    mod_id = db.insert(full_json, db.get_models(), upsert=overwrite)

    if generator is True:  # pragma: no cover
        full_json_data = {'mod_data_id': hexdi_m + data_hash,
                          'data_id': data_hash,
                          'data': data,
                          'data_val': data_val}

        db.insert(full_json_data, db.get_generators(), upsert=overwrite)

    try:
        results, res_dict = cm.train_pipe(train, save_params, model, data,
                                          data_val, generator, params_dump,
                                          data_hash, hexdi_m,
                                          *args, **kwargs)

        db.update({"_id": mod_id}, {'$set': res_dict})

    except Exception:
        db.update({"_id": mod_id}, {'$set': {'error': 1}})
        raise
    return results


@app.task(queue='sklearn')
def predict(model, data, *args, **kwargs):
    """Make predictions given a model and data

    Args:
        model (dict) : a serialied sklearn model.
        data(list, dict, np.array): data to be passed as a dictionary mapping
            inputs names to np.arrays or a list of arrays or an arrays

    Returns:
        an np.array of predictions
    """
    custom_objects = kwargs.get('custom_objects')

    # check if the predict function is already compiled
    m_id = model['mod_id']
    if m_id in COMPILED_MODELS:  # pragma: no cover
        model_instance = COMPILED_MODELS[m_id]['model']
        # load the attributes
        model_instance = load_params(model_instance, model['params_dump'])

    else:
        # get the model type
        model_dict = model['model_arch']

        # load model
        model_instance = model_from_dict_w_opt(model_dict,
                                               custom_objects=custom_objects)

        # load the attributes
        model_instance = load_params(model_instance, model['params_dump'])

        # write in the compiled list
        COMPILED_MODELS[m_id] = dict()
        COMPILED_MODELS[m_id]['model'] = model_instance

    # to be discussed
    # data = data[0]['X']

    return model_instance.predict(data)
