"""
.. codeauthor:: Thomas Boquet thomas.boquet@r2.ca

A simple module to perform training and prediction of models
------------------------------------------------------------------

Using `celery <http://www.celeryproject.org/>`_, this module helps to schedule
the training of models if the users send enough models in a short
period of time.

Serialization
~~~~~~~~~~~~~

The models are serialized in a json format in the database and loaded by a
worker when they are sent. The data is loaded via a request mapped in a
dictionnary.

.. note:

    The syntax will evolve a lot and changes have to be expected.

Compilation & cache
~~~~~~~~~~~~~~~~~~~

The models are compiled on the fly after the build. If the model is already
compiled and in the `COMPILED_MODEL` dictionnary mapping the models id to the
in memory compiled function, this function is used instead.

----------------------------------------------------------------------------
"""

import copy
import sys

from ..databasecon import get_models
from .utils import init_backend
from .utils import switch_backend


class Experiment(object):
    """An Experience train, predict, save and log a model

    Attributes:
        backend(str): the backend to use
        model_dict(dict, optionnal): the model to experience with"""
    backend = None

    def __init__(self, backend, model=None, metrics=None):
        backend, backend_name, backend_version = init_backend(backend)
        self.backend = backend
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.metrics = metrics
        self.model = model
        self.model_dict = self.backend.to_dict_w_opt(self.model, self.metrics)
        self.trained = False

    @property
    def model_dict(self):
        return self.__model_dict

    @model_dict.setter
    def model_dict(self, model_dict):
        if isinstance(model_dict, dict):
            self.__model_dict = dict()
            self.__model_dict['model_arch'] = model_dict
            self.__model_dict['mod_id'] = None
            self.__model_dict['params_dump'] = None
            self.__model_dict['data_id'] = None
        else:
            self.model = model_dict
            self.__model_dict['model_arch'] = self.backend.to_dict_w_opt(
                self.model, self.metrics)
            self.__model_dict['mod_id'] = None
            self.__model_dict['data_id'] = None
            self.__model_dict['params_dump'] = None

    @property
    def params_dump(self):
        return self.__params_dump

    @params_dump.setter
    def params_dump(self, params_dump):
        self.__model_dict['params_dump'] = params_dump
        self.__params_dump = params_dump

    @property
    def mod_id(self):
        return self.__mod_id

    @mod_id.setter
    def mod_id(self, mod_id):
        self.__model_dict['mod_id'] = mod_id
        self.__mod_id = mod_id

    @property
    def data_id(self):
        return self.__data_id

    @data_id.setter
    def data_id(self, data_id):
        self.__model_dict['data_id'] = data_id
        self.__data_id = data_id

    def fit(self, data, data_val, model=None, async=False, *args, **kwargs):
        """Build and fit a model given data and hyperparameters

        Args:
            data(list(dict)): a list of dictionnaries mapping inputs and
                outputs names to numpy arrays for training.
            data_val(list(dict)): a list of dictionnaries mapping inputs and
                outputs names to numpy arrays for validation.
            model(model, optionnal): a model from a supported backend

        Returns:
            the id of the model in the db, the id of the data in the db and
            path to the parameters.
        """
        _recompile = False
        if model is not None:
            self.model = model
            _recompile = True
        if "metrics" in kwargs:
            self.metrics = kwargs.pop("metrics")
            _recompile = True

        if _recompile is True:
            self.model_dict = self.backend.to_dict_w_opt(self.model,
                                                         self.metrics)

        res = self.backend.fit(self.backend_name, self.backend_version,
                               copy.deepcopy(self.model_dict), data,
                               data_val, *args, **kwargs)
        self.mod_id = res[0]
        self.data_id = res[1]
        self.params_dump = res[2]

        self.trained = True
        self.res = res

        return self.res

    def fit_async(self, data, data_val, model=None, async=False,
                  *args, **kwargs):
        """Build and fit asynchronously a model given data and hyperparameters

        Args:
            data(list(dict)): a list of dictionnaries mapping inputs and
                outputs names to numpy arrays for training.
            data_val(list(dict)): a list of dictionnaries mapping inputs and
                outputs names to numpy arrays for validation.
            model(model, optionnal): a model from a supported backend

        Returns:
            the id of the model in the db, the id of the data in the db and
            path to the parameters.
        """
        _recompile = False
        if model is not None:
            self.model = model
            _recompile = True
        if "metrics" in kwargs:
            self.metrics = kwargs.pop("metrics")
            _recompile = True

        if _recompile is True:
            self.model_dict = self.backend.to_dict_w_opt(self.model,
                                                         self.metrics)

        kwargs = self._check_serialize(kwargs)
        res = self.backend.fit.delay(self.backend_name, self.backend_version,
                                     copy.deepcopy(self.model_dict), data,
                                     data_val, *args, **kwargs)
        self.mod_id = res[0]
        self.data_id = res[1]
        self.params_dump = res[2]

        self.trained = True
        self.res = res

        return True

    def load_model(self, mod_id, data_id):
        self.mod_id = mod_id
        self.data_id = data_id
        models = get_models()
        model_db = models.find_one({'mod_id': mod_id, 'data_id': data_id})
        self._switch_backend(model_db)
        self.model_dict = model_db['model_arch']
        self.params_dump = model_db['params_dump']
        self.trained = True

    def _switch_backend(self, model_db):
        if model_db['backend_name'] != self.backend_name:
            backend = switch_backend(model_db['backend_name'])
            self.backend_name = backend.__name__
            self.backend_version = None
            if hasattr(backend, '__version__'):
                check = self.backend_version != backend.__version__
                self.backend_version = backend.__version__
            if check:  # pragma: no cover
                sys.stderr.write('Warning: the backend versions'
                                 'do not match.\n')  # pragma: no cover

    def predict(self, data):
        """Make predictions given data"""
        if self.trained:
            return self.backend.predict(self.model_dict, data)
        else:
            raise Exception("You must have a trained model"
                            "in order to make predictions")

    def _check_serialize(self, kwargs):
        for k in kwargs:
            if k in self.backend.TO_SERIALIZE:
                kwargs[k] = self.backend.serialize(kwargs[k])
        return kwargs
