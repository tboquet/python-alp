"""Tests for the sklearn backend"""


import numpy as np
import pytest
import sklearn
from sklearn import cross_validation
from sklearn import datasets
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

from alp.appcom.core import Experiment
from alp.backend import sklearn_backend as SKB

np.random.seed(1336)
NAME = sklearn.__name__
VERSION = sklearn.__version__

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                iris.data, iris.target, test_size=0.2, random_state=0)


def test_fit_predict_LinearRegression_normalizeF():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn linear regression model where normalization is set
         to False.
        NB: this is only for testing purpose. One should not try to predict
        a categorical variable with a linear regressor.
    """

    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    lr = LinearRegression(normalize=False)

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(lr)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=lr)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    lr.fit(X_train, y_train)
    assert np.allclose(predexp, lr.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, lr.predict(X_train))


def test_fit_predict_LinearRegression_normalizeT():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn linear regression model where normalization is set
         to True.
        NB: this is only for testing purpose. One should not try to predict
        a categorical variable with a linear regressor.
    """

    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    lr = LinearRegression(normalize=True)

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(lr)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=lr)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    lr.fit(X_train, y_train)
    assert np.allclose(predexp, lr.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, lr.predict(X_train))


def test_fit_LogisticRegression():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn logistic regression model.

        NB: by default, multi-class is set to OvR, eg one classifier per class.
        On the iris dataset, it means 3 classifiers.
        The attributes _ coef and intercept_, of shape (3,4) and (3,1) resp are
        serialized as intended.
    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    lr = LogisticRegression()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(lr)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=lr)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    lr.fit(X_train, y_train)
    assert np.allclose(predexp, lr.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, lr.predict(X_train))


def test_fit_OrthogonalMatchingPursuit():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn omp model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    omp = OrthogonalMatchingPursuit()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(omp)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=omp)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    omp.fit(X_train, y_train)
    assert np.allclose(predexp, omp.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, omp.predict(X_train))


def test_fit_Ridge():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn Ridge model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    ridge = Ridge()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(ridge)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=ridge)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    ridge.fit(X_train, y_train)
    assert np.allclose(predexp, ridge.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, ridge.predict(X_train))


def test_fit_KernelRidge():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn KernelRidge model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    kridge = KernelRidge()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(kridge)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=kridge)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    kridge.fit(X_train, y_train)
    assert np.allclose(predexp, kridge.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, kridge.predict(X_train))


def test_fit_BayesianRidge():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn BayesianRidge model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    bridge = BayesianRidge()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(bridge)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=bridge)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    bridge.fit(X_train, y_train)
    assert np.allclose(predexp, bridge.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, bridge.predict(X_train))


def test_fit_LassoLars():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn LassoLars model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    ll = LassoLars()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(ll)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=ll)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    ll.fit(X_train, y_train)
    assert np.allclose(predexp, ll.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, ll.predict(X_train))


def test_fit_Lars():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn Lars model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    l = Lars()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(l)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=l)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    l.fit(X_train, y_train)
    assert np.allclose(predexp, l.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, l.predict(X_train))


def test_fit_Lasso():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn Lasso model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    lasso = Lasso()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(lasso)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=lasso)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    lasso.fit(X_train, y_train)
    assert np.allclose(predexp, lasso.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, lasso.predict(X_train))


def test_fit_ARDRegression():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn ardr model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    ardr = ARDRegression()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(ardr)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=ardr)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    ardr.fit(X_train, y_train)
    assert np.allclose(predexp, ardr.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, ardr.predict(X_train))


def test_fit_QuadraticDiscriminantAnalysis():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn QuadraticDiscriminantAnalysis model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    qda = QuadraticDiscriminantAnalysis()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(qda)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=qda)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    qda.fit(X_train, y_train)
    assert np.allclose(predexp, qda.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, qda.predict(X_train))



def test_fit_LinearDiscriminantAnalysis():
    """Tests:
        - the to_dict_w_opt method of SKB (serialization);
        - the model_from_dict_w_opt of SKB (deserialization);
        - the fit method of SKB;
        - the fit method of the Experiment;
        - the predict method of the Experiment;
        - the predict method of the Experiment when loading from compiled;
        with a sklearn LinearDiscriminantAnalysis model.

    """
    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

    lda = LinearDiscriminantAnalysis()

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(lda)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=lda)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    lda.fit(X_train, y_train)
    assert np.allclose(predexp, lda.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, lda.predict(X_train))


if __name__ == "__main__":
    pytest.main([__file__])
