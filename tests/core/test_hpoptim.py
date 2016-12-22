"""Tests Hyper parameter search"""

import keras
import numpy as np
import pytest

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data
from sklearn.linear_model import LogisticRegression

from alp.appcom.core import Experiment
from alp.appcom.ensembles import HParamsSearch
from alp.appcom.utils import to_fuel_h5
from alp.utils.utils_tests import batch_size
from alp.utils.utils_tests import close_gens
from alp.utils.utils_tests import make_data
from alp.utils.utils_tests import make_gen
from alp.utils.utils_tests import sequential
from alp.utils.utils_tests import test_samples
from alp.utils.utils_tests import train_samples


def make_experiments():
    experiments = []
    nb_hiddens = [8, 16, 32]
    for i in range(3):
        nb_hidden = nb_hiddens[i]

        # model
        model = sequential(False)

        model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        expe = Experiment(model, metrics=['accuracy'])

        experiments.append(expe)
    return experiments


def make_sklearn_experiments():
    experiments = []
    C_list = [1.0, 0.8, 0.5]
    for C in C_list:
        model = LogisticRegression(C=C)
        expe = Experiment(model)
        experiments.append(expe)
    return experiments


class TestHParamsSearch:
    def test_fit(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss', op=np.min)
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)
        print(self)

    def test_fit_async(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss', op=np.min)
        param_search.fit_async([data], [data_val], nb_epoch=2,
                               batch_size=batch_size, verbose=2,
                               overwrite=True)
        param_search.summary(metrics={'val_loss': np.min})
        print(self)

    def test_fit_gen(self):
        gen, data, data_stream = make_gen(batch_size)
        val, data_2, data_stream_2 = make_gen(batch_size)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss', op=np.min)
        param_search.fit_gen([gen], [val], nb_epoch=2,
                             verbose=2,
                             nb_val_samples=128,
                             samples_per_epoch=64,
                             overwrite=True)
        param_search.summary(verbose=True, metrics={'val_loss': np.min})
        close_gens(gen, data, data_stream)
        close_gens(val, data_2, data_stream_2)
        print(self)

    def test_fit_gen_async(self):
        gen, data, data_stream = make_gen(batch_size)
        val, data_2, data_stream_2 = make_gen(batch_size)
        experiments = make_experiments()
        param_search = HParamsSearch(experiments, metric='loss', op=np.min)
        param_search.fit_gen_async([gen], [val], nb_epoch=2,
                                   verbose=2,
                                   nb_val_samples=128,
                                   samples_per_epoch=64,
                                   overwrite=True)
        param_search.summary(verbose=True, metrics={'val_loss': np.min})
        close_gens(gen, data, data_stream)
        close_gens(val, data_2, data_stream_2)
        print(self)

    def test_predict(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='acc', op=np.min)
        min_x = np.min(data['X'])
        data['X'] = (data['X'] - min_x) / (np.max(data['X']) - min_x)
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)

        param_search.predict(data['X'], metric='val_acc', op=np.max)

        experiments = make_experiments()
        param_search = HParamsSearch(experiments)
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)

        param_search.predict(data['X'], metric='acc', op=np.min)
        print(self)

    def test_predict_sklearn(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_sklearn_experiments()

        param_search = HParamsSearch(experiments, metric='score', op=np.max)
        data['y'] = np.argmax(data['y'], axis=1).ravel()
        data_val['y'] = np.argmax(data_val['y'], axis=1).ravel()
        param_search.fit([data], [data_val], overwrite=True)

        param_search.predict(data['X'])
        print(self)


if __name__ == "__main__":
    pytest.main([__file__])
