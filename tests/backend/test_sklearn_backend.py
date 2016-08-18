"""Tests for the sklearn backend"""


import numpy as np
import pytest
import sklearn
from sklearn import cross_validation
from sklearn import datasets

from alp.appcom.core import Experiment
from alp.backend import sklearn_backend as SKB

np.random.seed(1336)
NAME = sklearn.__name__
VERSION = sklearn.__version__

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                iris.data, iris.target, test_size=0.2, random_state=0)


data, data_val = dict(), dict()

data["X"] = X_train
data["y"] = y_train

data_val["X"] = X_test
data_val["y"] = y_test


keyval = dict()
for m in SKB.SUPPORTED:
    keyval[str(type(m()))[8:][:-2]] = m


@pytest.fixture(scope='module', params=list(keyval.keys()))
def get_model(request):
    return keyval[request.param]


class TestExperiment:
    def test_experiment_instance_utils(self, get_model):
        model = get_model()

        expe = Experiment(model)
        expe.model_dict = model
        expe.backend_name = 'another_backend'
        expe.model_dict = model
        print(self)

        assert expe.backend is not None

    def test_experiment_fit(self, get_model):
        model = get_model()

        expe = Experiment(model)

        for mod in [None, model]:
            expe.fit([data], [data_val], model=mod, overwrite=True)

        expe.backend_name = 'another_backend'
        expe.load_model()
        expe.load_model(expe.mod_id, expe.data_id)

        assert expe.data_id is not None
        assert expe.mod_id is not None
        assert expe.params_dump is not None
        print(self)

    def test_experiment_fit_async(self, get_model):
        model = get_model()

        expe = Experiment(model)

        for mod in [None, model]:
            expe.fit_async([data], [data_val], model=mod, overwrite=True)

        print(self)

    def test_experiment_predict(self, get_model):
        model = get_model()

        model._test_ = 'test'

        expe = Experiment(model)

        for mod in [None, model]:
            expe.fit([data], [data_val], model=mod, custom_objects={},
                           overwrite=True)
        expe.load_model()
        alp_pred = expe.predict(data['X'])

        model.fit(X_train, y_train)

        assert(np.allclose(alp_pred, model.predict(data['X'])))
        print(self)


def test_utils():
    objects = [list(),
                [1,2],
                [1.,2.],
                list(np.array([1, 2], dtype=np.integer)),
                list(np.array([1., 2.], dtype=np.float)),
                list(np.array([np.ones((1))]))]
    for el in objects:
        SKB.typeconversion(el)


if __name__ == "__main__":
    pytest.main([__file__])
