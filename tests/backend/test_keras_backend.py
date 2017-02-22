"""Tests for Keras backend"""

import copy

import inspect
import keras
import keras.backend as K
import numpy as np
import pytest
import six

import alp.appcom.utils as utls
from alp.appcom.core import Experiment
from alp.appcom.utils import switch_backend
from alp.backend import keras_backend as KTB
from alp.backend.common import open_dataset_gen
from alp.backend.keras_backend import get_function_name
from alp.backend.keras_backend import model_from_dict_w_opt
from alp.backend.keras_backend import serialize
from alp.backend.keras_backend import to_dict_w_opt
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


NAME = keras.__name__
VERSION = keras.__version__


def new_session():
    if K.backend() == 'tensorflow':  # pragma: no cover
        import tensorflow as tf
        K.clear_session()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        K.set_session(session)


new_session()


def prepare_model(get_model, get_loss_metric, custom):
    model = get_model

    loss, metric = get_loss_metric

    cust_objects = dict()

    if isinstance(metric, six.string_types):
        metrics = [metric]
    else:
        cust_objects[metric.__name__] = metric
        metrics = [metric()]

    if not isinstance(loss, six.string_types):
        cust_objects['cat_cross'] = loss
        if isinstance(loss, list):
            cust_objects['cat_cross'] = loss[-1]

        if model.__class__.__name__ == 'Model':
            loss_replace = loss
            if isinstance(loss, list):
                loss_replace = loss[-1]
            loss = dict()
            loss['main_loss'] = loss_replace

    if custom:
        cust_objects['Dropout_cust'] = return_custom

    if isinstance(loss, list):
        loss[-1] = loss[-1]()
    elif inspect.isfunction(loss):
        loss = loss()
    elif isinstance(loss, dict):
        loss = {k: v() for k, v in loss.items()}

    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=metrics,
                  custom_objects=cust_objects)
    return model, metrics, cust_objects


@pytest.fixture(scope='module',
                params=['one to many', 'many to one', 'many to many', 'val 1'])
def get_generators(request):
    if request.param == 'one to many':
        nb_train, nb_val = 4, 48
        gen_t, data_t, d_stream_t = make_gen(batch_size, nb_train)
        gen, data, d_stream = make_gen(batch_size, nb_val)

    elif request.param == 'many to one':
        nb_train, nb_val = 48, 4
        gen_t, data_t, d_stream_t = make_gen(batch_size, nb_train)
        gen, data, d_stream = make_gen(batch_size, nb_val)

    elif request.param == 'val 1':
        nb_train, nb_val = 4, 4
        gen_t, data_t, d_stream_t = make_gen(batch_size, nb_train)
        gen, data, d_stream = make_gen(batch_size, nb_val)

    elif request.param == 'many to many':
        nb_train, nb_val = 48, 48
        gen_t, data_t, d_stream_t = make_gen(batch_size, nb_train)
        gen, data, d_stream = make_gen(batch_size, nb_val)

    return gen_t, data_t, d_stream_t, gen, data, d_stream, (nb_train, nb_val)


@pytest.fixture
def get_loss():
    def return_loss():
        import keras.backend as K
        def cat_cross(y_true, y_pred):
            '''A test of custom loss function
            '''
            return K.categorical_crossentropy(y_pred, y_true)
        return cat_cross
    return return_loss


@pytest.fixture
def get_callback():
    def return_callback():
        from keras.callbacks import ReduceLROnPlateau
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)
        return reduce_lr
    return return_callback


@pytest.fixture
def get_metric():
    def return_metric():
        import keras.backend as K
        def cosine_proximity(y_true, y_pred):
            y_true = K.l2_normalize(y_true, axis=-1)
            y_pred = K.l2_normalize(y_pred, axis=-1)
            return -K.mean(y_true * y_pred)
        return cosine_proximity
    return return_metric


@pytest.fixture(scope='module', params=['sequential', 'model'])
def get_model(request):
    if request.param == 'sequential':
        return sequential
    elif request.param == 'model':
        return model
    print(self)


class TestExperiment:
    @pytest.fixture(params=['callback', 'no_callback'])
    def get_callback_fix(self, request):
        if request.param == 'callback':
            return [get_callback()]
        elif request.param == 'no_callback':
            return []

    @pytest.fixture(params=['classic', 'custom', 'list'])
    def get_loss_metric(self, request):
        if request.param == 'classic':
            return 'categorical_crossentropy', 'accuracy'
        elif request.param == 'custom':
            return get_loss(), get_metric()
        elif request.param == 'list':
            return [get_loss()], get_metric()
        print(self)

    @pytest.fixture(params=['c_layer', ''])
    def get_custom_l(self, request):
        if request.param == 'c_layer':
            return True
        elif request.param == '':
            return False
        print(self)

    def test_experiment_instance_utils(self, get_model):
        new_session()
        model = get_model()

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        expe = Experiment(model)
        expe.model_dict = model
        expe.backend_name = 'another_backend'
        expe.model_dict = model

        assert expe.backend is not None
        expe = Experiment()

        print(self)

    def test_experiment_fit(self, get_model, get_loss_metric,
                            get_custom_l, get_callback_fix):
        new_session()
        data, data_val = make_data(train_samples, test_samples)
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        expe = Experiment(model)

        for mod in [None, model]:
            for data_val_loc in [None, data_val]:
                expe.fit([data], [data_val_loc], model=mod, nb_epoch=2,
                         batch_size=batch_size, metrics=metrics,
                         custom_objects=cust_objects, overwrite=True,
                         callbacks=get_callback_fix)

        expe.backend_name = 'another_backend'
        expe.load_model()
        expe.load_model(expe.mod_id, expe.data_id)

        assert expe.data_id is not None
        assert expe.mod_id is not None
        assert expe.params_dump is not None

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_experiment_fit_async(self, get_model, get_loss_metric,
                                  get_custom_l):
        new_session()
        data, data_val = make_data(train_samples, test_samples)
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        cust_objects['test_list'] = [1, 2]
        expe = Experiment(model)

        expected_value = 2
        for mod in [None, model]:
            for data_val_loc in [None, data_val]:
                _, thread = expe.fit_async([data], [data_val_loc],
                                           model=mod, nb_epoch=2,
                                           batch_size=batch_size,
                                           metrics=metrics,
                                           custom_objects=cust_objects,
                                           overwrite=True,
                                           verbose=2)

                thread.join()

                for k in expe.full_res['metrics']:
                    if 'iter' not in k:
                        assert len(
                            expe.full_res['metrics'][k]) == expected_value

                if data_val_loc is not None:
                    for k in expe.full_res['metrics']:
                        if 'val' in k and 'iter' not in k:
                            assert None not in expe.full_res['metrics'][k]
                else:
                    for k in expe.full_res['metrics']:
                        if 'val' in k and 'iter' not in k:
                            assert all([np.isnan(v)
                                        for v in expe.full_res['metrics'][k]])

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_experiment_fit_gen(self, get_model, get_loss_metric,
                                get_custom_l):
        new_session()
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        model_name = model.__class__.__name__
        _, data_val_use = make_data(train_samples, test_samples)
        expe = Experiment(model)

        for val in [1, data_val_use]:
            gen, data, data_stream = make_gen(batch_size)
            if val == 1:
                val, data_2, data_stream_2 = make_gen(batch_size)
            expe.fit_gen([gen], [val], nb_epoch=2,
                         model=model,
                         metrics=metrics,
                         custom_objects=cust_objects,
                         samples_per_epoch=64,
                         nb_val_samples=128,
                         verbose=2, overwrite=True)

            close_gens(gen, data, data_stream)
            if val == 1:
                close_gens(val, data_2, data_stream_2)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_experiment_fit_gen_async(self, get_model, get_loss_metric,
                                      get_custom_l):
        new_session()
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        _, data_val_use = make_data(train_samples, test_samples)
        expe = Experiment(model)

        expected_value = 2
        for val in [None, 1, data_val_use]:
            gen, data, data_stream = make_gen(batch_size)
            if val == 1:
                val, data_2, data_stream_2 = make_gen(batch_size)
            _, thread = expe.fit_gen_async([gen], [val], nb_epoch=2,
                                           model=model,
                                           metrics=metrics,
                                           custom_objects=cust_objects,
                                           samples_per_epoch=64,
                                           nb_val_samples=128,
                                           verbose=2, overwrite=True)

            thread.join()

            for k in expe.full_res['metrics']:
                if 'iter' not in k:
                    assert len(
                        expe.full_res['metrics'][k]) == expected_value

            close_gens(gen, data, data_stream)
            if val == 1:
                close_gens(val, data_2, data_stream_2)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_experiment_generator_setups(self, get_generators):
        gen_t, data_t, d_stream_t, gen, data, d_stream, nb = get_generators
        nb_train, nb_val = nb
        test_model = model()

        test_model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop')
        expe = Experiment(test_model)
        expe.fit_gen([gen_t], [gen], nb_epoch=2,
                     samples_per_epoch=nb_train,
                     nb_val_samples=nb_val,
                     verbose=2, overwrite=True)
        close_gens(gen_t, data_t, d_stream_t)
        close_gens(gen, data, d_stream)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_experiment_predict(self, get_model, get_loss_metric):
        new_session()
        model, metrics, cust_objects = prepare_model(get_model(),
                                                     get_loss_metric,
                                                     False)

        model_name = model.__class__.__name__
        data, data_val = make_data(train_samples, test_samples)

        expe = Experiment(model)
        expe.fit([data], [data_val], nb_epoch=2,
                 batch_size=batch_size,
                 custom_objects=cust_objects,
                 metrics=metrics, overwrite=True)

        if model_name == 'Model':
            expe.predict({'X': data_val['X']})
        list_pred = expe.predict([data_val['X']])
        list_pred_2 = expe.predict(data_val['X'])

        assert len(list_pred) == len(data_val['X'])
        assert len(list_pred_2) == len(data_val['X'])

        print(self)

    def test_experiment_predict_async(self, get_model, get_loss_metric):
        new_session()
        model, metrics, cust_objects = prepare_model(get_model(),
                                                     get_loss_metric,
                                                     False)

        model_name = model.__class__.__name__
        data, data_val = make_data(train_samples, test_samples)

        expe = Experiment(model)
        expe.fit([data], [data_val], nb_epoch=2,
                 batch_size=batch_size,
                 custom_objects=cust_objects,
                 metrics=metrics, overwrite=True)

        if model_name == 'Model':
            expe.predict({'X': data_val['X']})
        async_pred = expe.predict_async([data_val['X']])
        async_pred_2 = expe.predict_async(data_val['X'])

        list_pred = async_pred.wait()
        list_pred_2 = async_pred_2.wait()

        assert len(list_pred) == len(data_val['X'])
        assert len(list_pred_2) == len(data_val['X'])

        print(self)

class TestBackendFunctions:
    def test_build_predict_func(self, get_model):
        """Test the build of a model"""
        new_session()
        X_tr = np.ones((train_samples, input_dim))
        model = get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model_name = model.__class__.__name__

        pred_func = KTB.build_predict_func(model)

        tensors = [X_tr]
        if model_name != 'Model':
            tensors.append(1.)

        res = pred_func(tensors)

        assert len(res[0]) == len(X_tr)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_fit(self, get_model):
        "Test the training of a serialized model"
        new_session()
        data, data_val = make_data(train_samples, test_samples)

        model = get_model()
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model_dict = dict()
        model_dict['model_arch'] = to_dict_w_opt(model)

        res = KTB.train(copy.deepcopy(model_dict['model_arch']), [data],
                        [data_val], [])
        res = KTB.fit(NAME, VERSION, model_dict, [data], 'test', [data_val],
                      [])

        assert len(res) == 4

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_predict(self, get_model):
        """Test to predict using the backend"""
        new_session()
        data, data_val = make_data(train_samples, test_samples)
        model = get_model()
        model.compile(optimizer='sgd', loss='categorical_crossentropy')

        expe = Experiment(model)
        expe.fit([data], [data_val])
        KTB.predict(expe.model_dict, [data['X']], False)
        KTB.predict(expe.model_dict, [data['X']], True)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)

    def test_serialization(self):
        model = sequential()
        to_dict_w_opt(model)
        print(self)

    def test_deserialization(self):
        new_session()
        model = sequential()
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        ser_mod = to_dict_w_opt(model)
        custom_objects = {'test_loss': [1, 2]}
        custom_objects = {k: serialize(custom_objects[k])
                          for k in custom_objects}
        model_from_dict_w_opt(ser_mod, custom_objects=custom_objects)

        if K.backend() == 'tensorflow':
            K.clear_session()

        print(self)


def test_utils():
    assert get_function_name("bob") == "bob"
    test_switch = switch_backend('sklearn')
    assert test_switch is not None
    gen, data, data_stream = make_gen(batch_size)
    open_dataset_gen(data_stream)
    gen.close()
    data.close(None)
    data_stream.close()
    for i in range(1, 20):
        utls.window(list(range(i * 2)), i)


if __name__ == "__main__":
    pytest.main([__file__])
