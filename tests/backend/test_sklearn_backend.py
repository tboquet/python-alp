"""Tests for the sklearn backend"""


import numpy as np
import pytest
import sklearn

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift

from six.moves import zip as szip

from sklearn import cross_validation as cv
from sklearn import datasets

from alp.appcom.core import Experiment
from alp.appcom.utils import to_fuel_h5
from alp.backend import sklearn_backend as SKB
from alp.backend.sklearn_backend import getname

np.random.seed(1336)
NAME = sklearn.__name__
VERSION = sklearn.__version__
CLASSIF = ['sklearn.linear_model.logistic.LogisticRegression',
           'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
           'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis']


def close_gens(gen, data, data_stream):
    gen.close()
    data.close(None)
    data_stream.close()


def generate_data(classif=False):
    data, data_val = dict(), dict()
    if classif:
        datas = datasets.load_iris()
        Xs = datas.data
        Ys = datas.target
    else:
        Xs = np.linspace(0, 12.3, num=150, endpoint=False).reshape(1, -1).T
        Ys = (Xs * np.sin(Xs)).ravel()

    data["X"], data_val["X"], data["y"], data_val["y"] = cv.train_test_split(
        Xs, Ys, test_size=20, random_state=0)

    return data, data_val


def dump_data(data, data_val, classif=False):
    '''
        The sklearn version differs from the keras version
            in the following points:
        - no local import of np
        - no graph model
        - validation cut at index 130
        - classification or regression data will dump different files
    '''
    suffix = '_R'
    if classif:
        suffix = '_C'

    inputs = [np.concatenate([data['X'], data_val['X']])]
    outputs = [np.concatenate([data['y'], data_val['y']])]

    file_name = 'test_data' + suffix
    scale = 1.0 / inputs[0].std(axis=0)
    shift = - scale * inputs[0].mean(axis=0)

    file_path, i_names, o_names = to_fuel_h5(inputs, outputs, [0, 130],
                                             ['train', 'test'],
                                             file_name,
                                             '/data_generator')
    return file_path, scale, shift, i_names, o_names


data_R, data_val_R = generate_data(False)
data_C, data_val_C = generate_data(True)

file_path_R, scale_R, shift_R, i_names_R, o_names_R = dump_data(
    data_R, data_val_R, False)
file_path_C, scale_C, shift_C, i_names_C, o_names_C = dump_data(
    data_C, data_val_C, True)


def make_gen(Nchunks=True, classif=False, train=True):
    '''
        Nchunks==True : 10 chunks in the generator
        Nchunks == False : 1 chunk in the generator
        Makes the distinction between classification/regression
        Makes the distinction between test/train
    '''

    file_path_f = file_path_R
    shift_f = shift_R
    scale_f = scale_R
    if classif:
        file_path_f = file_path_C
        shift_f = shift_C
        scale_f = scale_C

    if Nchunks:
        batch_size = 13
    else:
        batch_size = 130
    t_scheme = SequentialScheme(examples=130, batch_size=batch_size)
    t_source = 'train'
    if not train:
        if Nchunks:
            batch_size = 2
        else:
            batch_size = 20
        t_source = 'test'
        t_scheme = SequentialScheme(examples=20, batch_size=batch_size)

    t_set = H5PYDataset(file_path_f, which_sets=[t_source])
    data_stream_t = DataStream(dataset=t_set, iteration_scheme=t_scheme)

    stand_stream_t = ScaleAndShift(data_stream=data_stream_t,
                                   scale=scale_f, shift=shift_f,
                                   which_sources=t_source)

    return stand_stream_t, t_set, data_stream_t


keyval = dict()
for m in SKB.SUPPORTED:
    keyval[getname(m)] = m()


@pytest.fixture(scope='module', params=list(keyval.keys()))
def get_model_data_expe(request):
    model = keyval[request.param]
    expe = Experiment(model)

    data, data_val = data_R, data_val_R
    is_classif = False
    if getname(model, False) in CLASSIF:
        data, data_val = data_C, data_val_C
        is_classif = True

    return data, data_val, is_classif, model, expe


class TestExperiment:
    def test_experiment_instance_utils(self, get_model_data_expe):
        _, _, _, model, expe = get_model_data_expe
        expe.model_dict = model
        expe.backend_name = 'another_backend'
        expe.model_dict = model
        print(self)
        assert expe.backend is not None

    def test_experiment_fit(self, get_model_data_expe):
        data, data_val, _, model, expe = get_model_data_expe

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
        data, data_val, _, model, expe = get_model_data_expe

        for mod in [None, model]:
            expe.fit_async([data], [data_val], model=mod, overwrite=True)

        print(self)

    def test_experiment_predict(self, get_model_data_expe):
        data, data_val, _, model, expe = get_model_data_expe
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

    def test_experiment_fit_gen_nogenval(self, get_model_data_expe):
        '''
            Main case: gen on train, data on val
            Subcases:
                - 10 chunks on train B
                - 1 chunk on train B
        '''
        data, data_val, is_classif, model, expe = get_model_data_expe

        for Nchunks_gen, expected_value in szip([True, False], [10, 1]):
            gen_train, data_train, data_stream_train = make_gen(
                Nchunks_gen, is_classif, train=True)

            expe.fit_gen([gen_train], [data_val], overwrite=True)

            assert len(expe.full_res['metrics'][
                       'mean_absolute_error']) == expected_value
            assert len(expe.full_res['metrics'][
                       'val_mean_absolute_error']) == expected_value
            assert expe.data_id is not None
            assert expe.mod_id is not None
            assert expe.params_dump is not None
            assert expe

            close_gens(gen_train, data_train, data_stream_train)

        print(self)

    def test_experiment_fit_gen_withgenval(self, get_model_data_expe):
        '''
            Main case: gen on train, gen on val
            Subcases:
                - 10 chunks on train / 10 chunks on val C3
                - 10 chunks on train / 1 chunk on val C1
                - 1 chunk on train / 10 chunks on val C2
        '''
        data, data_val, is_classif, model, expe = get_model_data_expe

        for Nchunks_gen, Nchunks_val in szip([True, True, False],
                                             [True, False, True]):
            gen_train, data_train, data_stream_train = make_gen(
                Nchunks_gen, is_classif, train=True)
            gen_test, data_test, data_stream_test = make_gen(
                Nchunks_val, is_classif, train=False)

            expe.fit_gen([gen_train], [gen_test], overwrite=True)

            expected_value_gen = 10
            if not Nchunks_gen:
                expected_value_gen = 1

            assert len(expe.full_res['metrics'][
                       'mean_absolute_error']) == expected_value_gen
            assert len(expe.full_res['metrics'][
                       'val_mean_absolute_error']) == 10
            assert expe.data_id is not None
            assert expe.mod_id is not None
            assert expe.params_dump is not None
            assert expe

            close_gens(gen_train, data_train, data_stream_train)

        print(self)

    # def test_experiment_fit_gen_async_nogenval(self, get_model_data_expe):
    #     '''
    #         Main case: gen on train, data on val
    #         Subcases:
    #             - 10 chunks on train
    #             - 1 chunk on train
    #     '''
    #     data, data_val, is_classif, model, expe = get_model_data_expe

    #     for Nchunks_gen, expected_value in szip([True, False], [10, 1]):
    #         gen_train, data_train, data_stream_train = make_gen(
    #             Nchunks_gen, is_classif, train=True)

    #         res, thread = expe.fit_gen_async(
    #             [gen_train], [data_val], overwrite=True)
    #         thread.join()

    #         assert len(expe.full_res['metrics'][
    #                    'mean_absolute_error']) == expected_value
    #         assert len(expe.full_res['metrics'][
    #                    'val_mean_absolute_error']) == expected_value
    #         assert expe.data_id is not None
    #         assert expe.mod_id is not None
    #         assert expe.params_dump is not None
    #         assert expe

    #         close_gens(gen_train, data_train, data_stream_train)

    #     print(self)

    # def test_experiment_fit_gen_async_withgenval(self, get_model_data_expe):
    #     '''
    #         Main case: gen on train, gen on val
    #         Subcases:
    #             - 10 chunks on train / 10 chunks on val
    #             - 10 chunks on train / 1 chunk on val
    #             - 1 chunk on train / 10 chunks on val
    #     '''
    #     data, data_val, is_classif, model, expe = get_model_data_expe

    #     for Nchunks_gen, Nchunks_val in szip([True, True, False],
    #                                          [True, False, True]):

    #         gen_train, data_train, data_stream_train = make_gen(
    #             Nchunks_gen, is_classif, train=True)
    #         gen_test, data_test, data_stream_test = make_gen(
    #             Nchunks_val, is_classif, train=False)

    #         res, thread = expe.fit_gen_async(
    #             [gen_train], [gen_test], overwrite=True)
    #         thread.join()

    #         expected_value_gen = 10
    #         if not Nchunks_gen:
    #             expected_value_gen = 1

    #         assert len(expe.full_res['metrics'][
    #                    'mean_absolute_error']) == expected_value_gen
    #         assert len(expe.full_res['metrics'][
    #                    'val_mean_absolute_error']) == 10
    #         assert expe.data_id is not None
    #         assert expe.mod_id is not None
    #         assert expe.params_dump is not None
    #         assert expe

    #         close_gens(gen_train, data_train, data_stream_train)

    #     print(self)


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
