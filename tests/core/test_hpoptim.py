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

from alp.appcom.core import Experiment
from alp.appcom.ensembles import HParamsSearch
from alp.appcom.utils import to_fuel_h5
from alp.utils.utils_tests import batch_size
from alp.utils.utils_tests import close_gens
from alp.utils.utils_tests import input_dim
from alp.utils.utils_tests import make_data
from alp.utils.utils_tests import make_gen
from alp.utils.utils_tests import model
from alp.utils.utils_tests import return_custom
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


class TestHParamsSearch:
    def test_fit(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss')
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)
        print(self)

    def test_fit_async(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss')
        param_search.fit_async([data], [data_val], nb_epoch=2,
                               batch_size=batch_size, verbose=2,
                               overwrite=True)
        param_search.summary()
        print(self)

    def test_fit_gen(self):
        gen, data, data_stream = make_gen(batch_size)
        val, data_2, data_stream_2 = make_gen(batch_size)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss')
        param_search.fit_gen([gen], [val], nb_epoch=2,
                             verbose=2,
                             nb_val_samples=128,
                             samples_per_epoch=64,
                             overwrite=True)
        param_search.summary(verbose=True)
        close_gens(gen, data, data_stream)
        close_gens(val, data_2, data_stream_2)
        print(self)

    def test_fit_gen_async(self):
        gen, data, data_stream = make_gen(batch_size)
        val, data_2, data_stream_2 = make_gen(batch_size)
        experiments = make_experiments()
        param_search = HParamsSearch(experiments, metric='val_acc')
        param_search.fit_gen_async([gen], [val], nb_epoch=2,
                                   verbose=2,
                                   nb_val_samples=128,
                                   samples_per_epoch=64,
                                   overwrite=True)
        param_search.summary(verbose=True)
        close_gens(gen, data, data_stream)
        close_gens(val, data_2, data_stream_2)
        print(self)

    def test_predict(self):
        data, data_val = make_data(train_samples, test_samples)
        experiments = make_experiments()

        param_search = HParamsSearch(experiments, metric='loss')
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)

        param_search.predict(data['X'])

        experiments = make_experiments()
        param_search = HParamsSearch(experiments)
        param_search.fit([data], [data_val], nb_epoch=2,
                         batch_size=batch_size, verbose=2,
                         overwrite=True)

        param_search.predict(data['X'])
        print(self)


if __name__ == "__main__":
    pytest.main([__file__])
