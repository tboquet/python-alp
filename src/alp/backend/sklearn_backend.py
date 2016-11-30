"""
Adaptor for the sklearn backend
=============================
"""

import copy
import pickle
import re
import h5py
import numpy as np

from six import next as snext
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

SUPPORTED = [LogisticRegression, LinearRegression, Ridge, Lasso,
             Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge,
             ARDRegression, LinearDiscriminantAnalysis,
             QuadraticDiscriminantAnalysis, KernelRidge]


def getname(model, call=True):
    if call:
        m = model()
    else:
        m = model
    return(str(type(m))[8:][:-2])


keyval = dict()
for m in SUPPORTED:
    keyval[getname(m)] = m()

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
            dict_params[k] = v

    f = h5py.File(filepath, 'w')
    for k, v in dict_params.items():
        if v is not None:
            if type(v) is list:
                for i, val in enumerate(v):
                    kadd = "tolist" + str(i) + k
                    f[kadd] = val
            else:
                f[k] = v
        # so far the None case has been seen
        # only in Ridge when solver is not sag or lsqr.

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
    listed_params = dict()

    # first loop on f to get the parameters that are "unlisted"
    for k, v in f.items():
        if k[:6] == "tolist":
            listkeywithoutdigit = str(re.sub("\d+", "", k[6:]))
            digits = int(re.search(r'\d+', k[6:]).group())
            if listkeywithoutdigit not in listed_params.keys():
                listed_params[listkeywithoutdigit] = {digits: v}
            else:
                listed_params[listkeywithoutdigit][digits] = v

    # loop on listed_params that fills the temporary lists and set them in the
    # model
    for k, v in listed_params.items():
        lenlist = max(listed_params[k]) + 1
        stored = [None] * lenlist
        for i in range(lenlist):
            with listed_params[k][i].astype(listed_params[k][i].dtype):
                if listed_params[k][i].shape is not ():
                    stored[i] = listed_params[k][i][:]
                else:
                    out = listed_params[k][i][()]
        setattr(model, k, stored)

    # second loop on f.
    # TODO : merge the 2 loops on f.
    for k, v in f.items():
        if k[:6] != "tolist":
            with v.astype(v.dtype):
                if v.shape is not ():
                    out = v[:]
                else:
                    out = v[()]
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

    if isinstance(v, np.ndarray):  # pragma: no cover
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

    # to be discussed :
    # we add the metrics to the config even if it doesnt
    # make sense for a sklearn model
    # the metrics are then catch in model_from_dict_w_opt
    if metrics is not None:
        config['metrics'] = []
        for m in metrics:
            config['metrics'].append(m)

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

    # load the metrics
    if 'metrics' in model_dict:
        metrics = model_dict.pop('metrics')
    else:
        metrics = None

    # create a new instance of the appropriate model type
    model = copy.deepcopy(keyval[model_dict['config']])

    # load the parameters
    for k, v in model_dict.items():
        if isinstance(v, list):  # pragma: no cover
            setattr(model, k, np.array(v))
        else:
            setattr(model, k, v)

    return model, metrics


def train(model, data, data_val, size_gen, generator=False, *args, **kwargs):
    """Fit a model given parameters and a serialized model

    Args:
        model(dict): a serialized sklearn model
        data(list): a list of dict mapping inputs and outputs to lists or
            dictionnaries mapping the inputs names to np.arrays
            XOR -a list of fuel generators
        data_val(list): same structure than `data` but for validation.

        it is possible to feed generators for data and plain data for data_val.
        it is not possible the other way around.

    Returns:
        the loss (list), the validation loss (list), the number of iterations,
        and the model
        """

    # Local variables
    import sklearn.metrics

    results = dict()
    results['metrics'] = dict()
    custom_objects = None
    predondata = []
    predonval = []
    fit_gen_val = False

    # Load custom_objects
    if 'custom_objects' in kwargs:  # pragma: no cover
        custom_objects = kwargs.pop('custom_objects')

    # Load model and get metrics
    model, metrics = model_from_dict_w_opt(model,
                                           custom_objects=custom_objects)

    # instantiates metrics
    # there is at least one mandatory metric for sklearn models
    metrics_names = ["score"]
    if metrics:
        for metric in metrics:
            metrics_names.append(metric)
    for metric in metrics_names:
        results['metrics'][metric] = []
        results['metrics']["val_" + metric] = []

    # pickle data if generator
    if generator:
        data = [pickle.loads(d.encode('raw_unicode_escape')) for d in data]

    # check if data_val is in generator
    if all(v is None for v in data_val):
        val_gen = 0
    else:
        val_gen = check_gen(data_val)
    # if so pickle data_val
    if val_gen > 0:
        if generator:
            data_val = [pickle.loads(dv.encode('raw_unicode_escape'))
                        for dv in data_val]
            fit_gen_val = True
        else:
            raise Exception("You should also pass a generator for the training"
                            " data.")

    # Fit the model
    # and validates it
    if len(size_gen) == 0:
        size_gen = [0] * len(data)
    # loop over the data/generators
    for d, dv, s_gen in szip(data, data_val, size_gen):
        # check if we have a data_val object.
        # if not, no evaluation of the metrics on data_val.
        if dv is None:
            validation = False
        else:
            validation = True

        # not treating the case "not generator and fit_gen_val"
        #    since it is catched above
        # case A : dict for data and data_val
        if not generator and not fit_gen_val:
            X, y = d['X'], d['y']
            model.fit(X, y, *args, **kwargs)
            predondata.append(model.predict(X))
            for metric in metrics_names:
                if metric is not 'score':
                    computed_metric = getattr(
                        sklearn.metrics, metric)(y, predondata[-1])
                    results['metrics'][metric].append(
                        computed_metric)
                else:
                    computed_metric = model.score(X, y)
                    results['metrics']['score'].append(
                        computed_metric)
                    # TODO : optimization

            if validation:
                X_val, y_val = dv['X'], dv['y']
                predonval.append(model.predict(X_val))
                for metric in metrics_names:
                    if metric is not 'score':
                        computed_metric = getattr(
                            sklearn.metrics, metric)(y_val, predonval[-1])
                    else:
                        computed_metric = model.score(X, y)
                        # TODO : optimization
                    results['metrics']['val_' + metric].append(
                        computed_metric)

            else:
                for metric in metrics_names:
                    results['metrics']['val_' + metric].append(np.nan)

        # case B : generator for data and no generator for data_val
        # could be dict or None
        elif generator and not fit_gen_val:
            if validation:
                X_val, y_val = dv['X'], dv['y']
            for batch_data in d.get_epoch_iterator():
                X, y = batch_data
                model.fit(X, y, *args, **kwargs)
                predondata.append(model.predict(X))
                if validation:
                    predonval.append(model.predict(X_val))

                for metric in metrics_names:
                    if metric is not 'score':
                        results['metrics'][metric].append(
                            getattr(sklearn.metrics, metric)(y,
                                                             predondata[-1]))

                        if validation:
                            results['metrics']['val_' +
                                               metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y_val, predonval[-1]))
                        else:
                            results['metrics'][
                                'val_' + metric].append(np.nan)
                    else:
                        results['metrics']['score'].append(
                            model.score(X, y))
                        if validation:
                            results['metrics']['val_score'].append(
                                model.score(X_val, y_val))
                        else:
                            results['metrics']['val_score'].append(np.nan)

        # case C : generator for data and for data_val
        else:
            # case C1: N chunks in gen, 1 chunk in val, many to one
            if s_gen == 1:
                X_val, y_val = snext(dv.get_epoch_iterator())
                for batch_data in d.get_epoch_iterator():
                    X, y = batch_data
                    model.fit(X, y, *args, **kwargs)
                    predondata.append(model.predict(X))
                    predonval.append(model.predict(X_val))
                    for metric in metrics_names:
                        if metric is not 'score':
                            results['metrics'][metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y, predondata[-1]))
                            results['metrics']['val_' +
                                               metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y_val, predonval[-1]))
                        else:
                            results['metrics']['score'].append(
                                model.score(X, y))
                            results['metrics']['val_score'].append(
                                model.score(X_val, y_val))

            # case C2 : 1 chunk in gen, N chunks in val, one to many
            elif s_gen == 2:
                X, y = snext(d.get_epoch_iterator())
                model.fit(X, y, *args, **kwargs)
                predondata.append(model.predict(X))
                for metric in metrics_names:
                    if metric is not 'score':
                        results['metrics'][metric].append(
                            getattr(sklearn.metrics,
                                    metric)(y, predondata[-1]))
                    else:
                        results['metrics']['score'].append(model.score(X, y))

                for batch_val in dv.get_epoch_iterator():
                    X_val, y_val = batch_val
                    predonval.append(model.predict(X_val))
                    for metric in metrics_names:
                        if metric is not 'score':
                            results['metrics']['val_' +
                                               metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y_val, predonval[-1]))
                        else:
                            results['metrics']['val_score'].append(
                                model.score(X_val, y_val))

            # case C3 : same numbers of chunks, many to many
            elif s_gen == 3:
                for batch_data, batch_val in szip(d.get_epoch_iterator(),
                                                  dv.get_epoch_iterator()):
                    X, y = batch_data
                    X_val, y_val = batch_val
                    model.fit(X, y, *args, **kwargs)
                    predondata.append(model.predict(X))
                    predonval.append(model.predict(X_val))
                    for metric in metrics_names:
                        if metric is not 'score':
                            results['metrics'][metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y, predondata[-1]))
                            results['metrics']['val_' +
                                               metric].append(
                                getattr(sklearn.metrics,
                                        metric)(y_val, predonval[-1]))
                        else:
                            results['metrics']['score'].append(
                                model.score(X, y))
                            results['metrics']['val_score'].append(
                                model.score(X_val, y_val))

            else:  # pragma: no cover
                raise Exception(
                    'Incoherent generator size for train and validation')

    # for compatibility with keras backend
    results['metrics']['iter'] = np.nan

    return results, model


@app.task(bind=True, default_retry_delay=60 * 10, max_retries=3,
          rate_limit='120/m', queue='sklearn')
def fit(self, backend_name, backend_version, model, data, data_hash,
        data_val, size_gen, generator=False, *args, **kwargs):
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
                          'data': data}

        db.insert(full_json_data, db.get_generators(), upsert=overwrite)

    try:
        results, res_dict = cm.train_pipe(train, save_params, model,
                                          data, data_val,
                                          generator, size_gen,
                                          params_dump, data_hash,
                                          hexdi_m,
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
        model_instance, _ = model_from_dict_w_opt(
            model_dict,
            custom_objects=custom_objects)

        # load the attributes
        model_instance = load_params(model_instance, model['params_dump'])

        # write in the compiled list
        COMPILED_MODELS[m_id] = dict()
        COMPILED_MODELS[m_id]['model'] = model_instance

    # to be discussed
    # data = data[0]['X']

    return model_instance.predict(data)
