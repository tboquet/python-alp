"""
Adaptor for the sklearn backend
=============================
"""

import types

import dill
import six

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.kernel_ridge import KernelRidge


from ..celapp import app
from ..config import PATH_H5

SUPPORTED = [LogisticRegression, LinearRegression, Ridge, Lasso,
             Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge,
             ARDRegression, LinearDiscriminantAnalysis,
             QuadraticDiscriminantAnalysis, KernelRidge]

keyval = dict()
for m in SUPPORTED:
    keyval[str(type(m()))[8:][:-2]] = m()


COMPILED_MODELS = dict()
TO_SERIALIZE = ['custom_objects']
dill.settings['recurse'] = True


# general utilities

def get_backend():
    import sklearn as SK
    return SK


def serialize(custom_object):
    return dill.dumps(six.get_function_code(custom_object))


def deserialize(k, custom_object_str):
    code = dill.loads(custom_object_str)
    return types.FunctionType(code, globals(), k)


def save_params(model, filepath):

    attr = model.__dict__
    for k, v in attr.items():
        if k[-1:] == '_':
            dict_params[k] = typeconversion(v)

    f = h5py.File(filepath, 'w')
    for k, v in dict_params.items():
        if v is not None:
            f[k] = v
        else:
            # so far seen only in Ridge when solver is not sag or lsqr.
            pass

    f.flush()
    f.close()


def load_params(model, filepath):
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        with v.astype(v.dtype):
            if v.shape is not ():
                out = v[:]
            else:
                out = v
            setattr(model, k, out)
    return model


def typeconversion(v):

    if isinstance(v, np.ndarray):
        return v.tolist()

    elif isinstance(v, list):
        if len(v) == 0:
            return v
        else:
            if isinstance(v[0], np.integer):
                return [int(vv) for vv in v]
            elif isinstance(v[0], np.float):
                return [float(vv) for vv in v]
            elif isinstance(v[0], np.ndarray):
                return [vv.tolist() for vv in v]
            else:
                return v
    else:
        return v


def to_dict_w_opt(model, metrics=None):
    # ??? metrics ??

    # que faire du dict params ? solution proposée : on save weight après le
    # fit seulement

    modeltobedump = dict()
    # dict_params = dict()

    typestring = str(type(model))[8:][:-2]
    if verbose:
        print(typestring)
    modeltobedump['type'] = typestring

    attr = model.__dict__

    for k, v in attr.items():
        if k[-1:] == '_':
            pass
            # dict_params[k] = typeconversion(v)
        else:
            modeltobedump[k] = typeconversion(v)

    # Dumping and writing

    # smodel = json.dumps(modeltobedump)
    # return smodel, dict_params
    return modeltobedump

    # from keras: return config


def model_from_dict_w_opt(model_dict, custom_objects=None):

    if custom_objects is None:
        custom_objects = dict()

    custom_objects = {k: deserialize(k, custom_objects[k])
                      for k in custom_objects}

    # TODO : layer from config like
    # si le nom de custom_object est dans une clé de modellod.item on le met.

    modelload = model_dict
    if modelload['type'] not in keyval:
        raise NotImplementedError("Scikit model not supported.")

    model = copy.deepcopy(keyval[modelload['type']])

    for k, v in modelload.items():
        if isinstance(v, list):
            setattr(model, k, np.array(v))
        else:
            setattr(model, k, v)

    return model
    # model est une nouvelle instance d'un modèle sklearn supporté


# core utilities

# def build_predict_func(mod):
# ??? pas besoin
# return K.function(mod.inputs, mod.outputs, updates=mod.state_updates)


def train(model, data, data_val, *args, **kwargs):

    custom_objects = None
    metrics = []

    if 'custom_objects' in kwargs:
        custom_objects = kwargs.pop('custom_objects')

    if 'metrics' in kwargs:
        print(" metrics not supported.")
        # metrics = kwargs.pop('metrics') #

    loss = []
    val_loss = []
    # load model
    model = model_from_dict_w_opt(model, custom_objects=custom_objects)
    mod_name = model['type']

    # fit the model according to the input/output type
    # TODO : add check de bonne forme des data
    ldm = []
    predondata = []
    predonval = []

    for d, dv in zip(data, data_val):
        model.fit(d['X'], d['y'], *args, **kwargs)
        X_val, y_val = dv['X'], dv['y']
        predondata.append(model.predict(data[0]))
        predonval.append(model.predict(dv[0]))

    # TODO : métrique de validation !!!!
    from sklearn.metrics import mean_absolute_error
    metrics.append(mean_absolute_error)
    for metric in metrics:
        for d, dv, pda, pva in zip(data, data_val, predondata, predonval):
            loss.append(mean_absolute_error(d['X'], pda))
            val_loss.append(mean_absolute_error(dv['X'], pva))

    # ??? max iter ??
    max_iter = np.nan

    return loss, val_loss, max_iter, model


@app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
def fit(backend_name, backend_version, model, data, data_val, *args, **kwargs):

    from ..databasecon import get_models
    from datetime import datetime
    import hashlib
    import json
    import numpy as np

    # convert dict to json string
    model_str = json.dumps(model)

    # get the models collection
    models = get_models()  # ????

    first = list(data[0].keys())[0]
    un_data_m = data[0][first].mean()
    un_data_f = data[0][first][0]

    # create the model hash from the stringified json
    mh = hashlib.md5()
    str_concat_m = str(model_str)  # + str(kwargs['batch_size'])
    mh.update(str_concat_m.encode('utf-8'))
    hexdi_m = mh.hexdigest()

    # create the data hash
    dh = hashlib.md5()
    str_concat_d = str(un_data_m) + str(un_data_f)
    dh.update(str_concat_d.encode('utf-8'))
    hexdi_d = dh.hexdigest()

    params_dump = PATH_H5 + hexdi_m + hexdi_d + '.h5'

    # update the full json
    full_json = {'backend_name': backend_name,
                 'backend_version': backend_version,
                 'model_type': model['type'],  # changement par rapport à keras
                 'datetime': datetime.now(),
                 'mod_id': hexdi_m,
                 'data_id': hexdi_d,
                 'params_dump': params_dump,
                 # kerazs : 'batch_size': kwargs['batch_size'],
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
            'date_finished_training': datetime.now()
        }})

        save_params(model, filepath=params_dump)
        # keras : model.save_weights(params_dump, overwrite=True)

    except Exception:
        models.update({"_id": mod_id}, {'$set': {'error': 1}})
        raise
    return hexdi_m, hexdi_d, params_dump


@app.task
def predict(model, data, *args, **kwargs):

    custom_objects = kwargs.get('custom_objects')

    # check if the predict function is already compiled
    if model['mod_id'] in COMPILED_MODELS:
        model_sk = COMPILED_MODELS[model['mod_id']]['model']
        model_name = model['type']

    else:
        # get the model type
        model_dict = model['type']

        # model_dict.pop('optimizer')

        # load model
        model_skk = model_from_dict_w_opt(model_dict,
                                          custom_objects=custom_objects)

        # load the weights
        model_sk = load_params(model_k, model['params_dump'])

        # write in the compiled list
        COMPILED_MODELS[model['mod_id']] = dict()
        COMPILED_MODELS[model['mod_id']]['model'] = model_sk

    return model_sk.predict(data)
