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


data, data_val = dict(), dict()

data["X"] = X_train
data["y"] = y_train

data_val["X"] = X_test
data_val["y"] = y_test


def _test_fit_predict_model(imodel):

    model_dict = dict()
    model_dict['model_arch'] = SKB.to_dict_w_opt(imodel)
    model_deserialized = SKB.model_from_dict_w_opt(model_dict['model_arch'])
    assert model_deserialized is not None

    res = SKB.fit(NAME, VERSION, model_dict, [data], [data_val])
    assert len(res) == 3

    expe = Experiment("sklearn", model=imodel)
    assert expe.backend is not None

    expe.fit([data], [data_val])
    assert expe.data_id is not None
    assert expe.mod_id is not None
    assert expe.params_dump is not None

    predexp = expe.predict(data["X"])
    imodel.fit(X_train, y_train)
    assert np.allclose(predexp, imodel.predict(X_train))
    predexp = expe.predict(data["X"])
    assert np.allclose(predexp, imodel.predict(X_train))


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

    lr = LinearRegression(normalize=False)
    _test_fit_predict_model(lr)


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
    lr = LinearRegression(normalize=True)
    _test_fit_predict_model(lr)


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

    lr = LogisticRegression()
    _test_fit_predict_model(lr)


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

    omp = OrthogonalMatchingPursuit()
    _test_fit_predict_model(omp)


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

    ridge = Ridge()
    _test_fit_predict_model(ridge)


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

    kridge = KernelRidge()
    _test_fit_predict_model(kridge)


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

    bridge = BayesianRidge()
    _test_fit_predict_model(bridge)


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

    ll = LassoLars()
    _test_fit_predict_model(ll)


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

    l = Lars()
    _test_fit_predict_model(l)


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

    l = Lasso()
    _test_fit_predict_model(l)


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

    ardr = ARDRegression()
    _test_fit_predict_model(ardr)


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

    qda = QuadraticDiscriminantAnalysis()
    _test_fit_predict_model(qda)


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

    lda = LinearDiscriminantAnalysis()
    _test_fit_predict_model(lda)


if __name__ == "__main__":
    pytest.main([__file__])
