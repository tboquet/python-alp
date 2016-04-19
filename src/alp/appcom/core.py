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

from celery import Celery
from .utils import appbackend

COMPILED_MODELS = dict()


app = Celery(broker='amqp://guest:guest@rabbitmq:5672//',
             backend='mongodb://mongo_r:27017')


class Experience(object):
    """An Experience train, predict, save and log a model

    Attributes:
        backend(str): the backend to use
        model_dict(dict, optionnal): the model to experience with"""
    backend = None

    @appbackend
    def __init__(self, backend, model_dict=None):
        self.model_dict = model_dict

    def build(self, model_dict=None):
        """Build a model
        """
        if model_dict is not None:
            self.m_to_build = model_dict
        elif self.model_dict is not None:
            self.m_to_build = self.model_dict
        else:
            raise Exception("You must provide a model.")

        self.built_model = self.backend.build(self.m_to_build)

    @app.task(default_retry_delay=60 * 10, max_retries=3, rate_limit='120/m')
    def fit(self, data, params):
        if self.built_model:
            res = self.backend.fit(self.built_model, data, params)
            self.trained = True
        else:
            raise Exception("You nust have built a model.")

        return res

    @app.task
    def predict(self, data):
        if self.trained:
            return self.backend.predict(self.built_model, data)
