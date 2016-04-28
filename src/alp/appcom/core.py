"""
.. codeauthor:: Thomas Boquet thomas.boquet@r2.ca

A simple module to perform training and prediction of Keras models
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
compiled and in the `COMPILED_MODEL` dictionnary, mapping the models id to the
in memory compiled function, this function is used instead.

----------------------------------------------------------------------------
"""

from .utils import appbackend

COMPILED_MODELS = dict()


class Experiment(object):
    """An Experience train, predict, save and log a model

    Attributes:
        backend(str): the backend to use
        model_dict(dict, optionnal): the model to experience with"""
    backend = None

    @appbackend
    def __init__(self, backend, model=None, metrics=None):
        self.metrics = metrics
        self.model_dict = self.backend.to_dict_w_opt(model, self.metrics)
        self.trained = False

    def fit(self, data, data_val, model=None, *args, **kwargs):
        """Build and fit a model given data and hyperparameters"""
        if "metrics" in kwargs:
            self.metrics = kwargs.pop("metrics")
        if model is not None:
            self.model_dict = self.backend.to_dict_w_opt(model, self.metrics)
        if self.model_dict is None:
            raise Exception("You must pass a model to an Experiment")
        self.res = self.backend.fit(self.model_dict, data, data_val,
                                    *args, **kwargs)
        self.trained = True

        return self.res

    def predict(self, data):
        """Make predictions given data"""
        if self.trained:
            return self.backend.predict(self.model_dict, data)
        else:
            raise Exception("You must have a trained model"
                            "in order to make predictions")
