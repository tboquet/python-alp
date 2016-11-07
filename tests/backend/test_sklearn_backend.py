"""Tests for the sklearn backend"""


import numpy as np
import pytest
import sklearn
from sklearn import cross_validation as cv
from sklearn import datasets

from alp.appcom.core import Experiment
from alp.backend import sklearn_backend as SKB
from alp.backend.sklearn_backend import getname

np.random.seed(1336)
NAME = sklearn.__name__
VERSION = sklearn.__version__
CLASSIF = ['sklearn.linear_model.logistic.LogisticRegression',
           'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
           'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis']


def generate_data(classif=False):
    data, data_val = dict(), dict()
    if classif:
        datas = datasets.load_iris()
        Xs = datas.data
        Ys = datas.target
    else:
        Xs = np.linspace(-2, 10).reshape(1, -1).T
        Ys = (Xs * np.sin(Xs)).ravel()

    data["X"], data_val["X"], data["y"], data_val["y"] = cv.train_test_split(
        Xs, Ys, test_size=0.2, random_state=0)

    return data, data_val

data_R, data_val_R = generate_data(False)
data_C, data_val_C = generate_data(True)

keyval = dict()
for m in SKB.SUPPORTED:
    keyval[getname(m)] = m


@pytest.fixture(scope='module', params=list(keyval.keys()))
def get_model_data_expe(request):
    model = keyval[request.param]()
    expe = Experiment(model)

    data, data_val = data_R, data_val_R
    if getname(model, False) in CLASSIF:
            data, data_val = data_C, data_val_C

    return data, data_val, model, expe


class TestExperiment:
    def test_experiment_instance_utils(self, get_model_data_expe):
        _, _, model, expe = get_model_data_expe
        expe.model_dict = model
        expe.backend_name = 'another_backend'
        expe.model_dict = model
        print(self)

        assert expe.backend is not None

    def test_experiment_fit(self, get_model_data_expe):
        data, data_val, model, expe = get_model_data_expe

        for mod in [None, model]:
            expe.fit([data], [data_val], model=mod, overwrite=True)

        expe.backend_name = 'another_backend'
        expe.load_model()
        expe.load_model(expe.mod_id, expe.data_id)

        assert expe.data_id is not None
        assert expe.mod_id is not None
        assert expe.params_dump is not None
        print(self)

    def test_experiment_fit_async(self, get_model_data_expe):
        data, data_val, model, expe = get_model_data_expe

        for mod in [None, model]:
            expe.fit_async([data], [data_val], model=mod, overwrite=True)

        print(self)

    def test_experiment_predict(self, get_model_data_expe):
        data, data_val, model, expe = get_model_data_expe
        model._test_ = 'test'

        for mod in [None, model]:
            expe.fit([data], [data_val], model=mod, custom_objects={},
                     overwrite=True)
        expe.load_model()
        alp_pred = expe.predict(data['X'])

        model.fit(data['X'], data['y'])
        sklearn_pred = model.predict(data['X'])
        assert(np.allclose(alp_pred, sklearn_pred))
        print(self)


def test_utils():
    objects = [list(),
               [1, 2],
               [1., 2.],
               list(np.array([1, 2], dtype=np.integer)),
               list(np.array([1., 2.], dtype=np.float)),
               list(np.array([np.ones((1))]))]
    for el in objects:
        SKB.typeconversion(el)


if __name__ == "__main__":
    pytest.main([__file__])
