"""Tests for the Keras backend"""

import copy
import keras
import keras.backend as K
import numpy as np
import pytest
import six
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift
from keras.engine import Layer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Graph
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

from alp.appcom.core import Experiment
from alp.appcom.utils import imports
from alp.appcom.utils import switch_backend
from alp.appcom.utils import to_fuel_h5
from alp.backend import keras_backend as KTB
from alp.backend.common import open_dataset_gen
from alp.backend.keras_backend import get_function_name
from alp.backend.keras_backend import to_dict_w_opt

np.random.seed(1337)


input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 256
test_samples = 64
NAME = keras.__name__
VERSION = keras.__version__

def make_data():
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                                nb_test=test_samples,
                                                input_shape=(input_dim,),
                                                classification=True,
                                                nb_class=nb_class)

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)

    data, data_val = dict(), dict()

    data["X"] = X_tr
    data["y"] = y_tr

    data_val["X"] = X_te
    data_val["y"] = y_te
    return data, data_val


def dump_data(graph=False):
    data, data_val = make_data()
    inputs = [np.concatenate([data['X'], data_val['X']])]
    outputs = [np.concatenate([data['y'], data_val['y']])]

    file_name = 'test_data_'
    scale = 1.0 / inputs[0].std(axis=0)
    shift = - scale * inputs[0].mean(axis=0)
    if graph:
        inputs = {'X': np.concatenate([data['X'], data_val['X']])}
        outputs = {'y': np.concatenate([data['y'], data_val['y']])}
        file_name += "graph_"
        scale = 1.0 / inputs['X'].std(axis=0)
        shift = - scale * inputs['X'].mean(axis=0)

    file_path = to_fuel_h5(inputs, outputs, [0, 256], ['train', 'test'],
                           file_name,
                           '/data_generator')
    return file_path, scale, shift

file_path, scale, shift = dump_data()
file_path_g, scale_g, shift_g = dump_data(graph=True)


def make_gen(graph=False):
    file_path_f = file_path
    if graph:
        file_path_f = file_path_g
    train_set = H5PYDataset(file_path_f,
                            which_sets=('train','test'))

    scheme = SequentialScheme(examples=196, batch_size=32)

    data_stream_train = DataStream(dataset=train_set, iteration_scheme=scheme)

    stand_stream_train = ScaleAndShift(data_stream=data_stream_train,
                                       scale=scale, shift=shift,
                                       which_sources=('input_X',))
    return stand_stream_train, train_set, data_stream_train


class Dropout_cust(Layer):
    '''Applies Dropout to the input.
    '''
    def __init__(self, p, **kwargs):
        self.p = p
        if 0. < self.p < 1.:
            self.uses_learning_phase = True
        self.supports_masking = True
        super(Dropout_cust, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if 0. < self.p < 1.:
            x = K.in_train_phase(K.dropout(x, level=self.p), x)
        return x

    def get_config(self):
        config = {'p': self.p}
        base_config = super(Dropout_cust, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sequential(custom=False):
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.add(Dropout(0.5))
    if custom:
        model.add(Dropout_cust(0.5))
    return model


def model(custom=False):
    inputs = Input(shape=(input_dim,), name='X')

    x = Dense(nb_hidden, activation='relu')(inputs)
    x = Dense(nb_hidden, activation='relu')(x)
    predictions = Dense(nb_class,
                        activation='softmax',
                        name='main_loss')(x)

    model = Model(input=inputs, output=predictions)
    return model


def graph(custom=False):
    name='dense1'
    model = Graph()
    model.add_input(name='X', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                name='dense1', input='X')
    if custom:
        name = 'do'
        model.add_node(Dropout_cust(0.5), name=name, input='dense1')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input=name)
    model.add_node(Dropout(0.5), name='last_dropout', input='last_dense')

    model.add_output(name='y', input='last_dropout')
    return model


def prepare_model(get_model, get_loss_metric, custom):
    model = get_model

    loss, metric = get_loss_metric

    cust_objects = dict()

    metrics = [metric]

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
        cust_objects['Dropout_cust'] = Dropout_cust


    model.compile(loss=loss,
                  optimizer='rmsprop',
                  metrics=metrics,
                  custom_objects=cust_objects)
    return model, metrics, cust_objects


@pytest.fixture
def get_loss():
    @imports({"K": K})
    def cat_cross(y_true, y_pred):
        '''A test of custom loss function
        '''
        return K.categorical_crossentropy(y_pred, y_true)
    return cat_cross


@pytest.fixture
def get_metric():
    @imports({"K": K})
    def cosine_proximity(y_true, y_pred):
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return -K.mean(y_true * y_pred)
    return cosine_proximity


@pytest.fixture(scope='module', params=['sequential', 'model', 'graph'])
def get_model(request):
    if request.param == 'sequential':
        return sequential
    elif request.param == 'model':
        return model
    elif request.param == 'graph':
        return graph
    print(self)


class TestExperiment:
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
        model = get_model()

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        expe = Experiment(model)
        expe.model_dict = model
        expe.backend_name = 'another_backend'
        expe.model_dict = model
        print(self)

        assert expe.backend is not None

    def test_experiment_fit(self, get_model, get_loss_metric,
                            get_custom_l):
        data, data_val = make_data()
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        expe = Experiment(model)

        if get_custom_l:
            with pytest.raises(RuntimeError) as excinfo:
                expe.fit([data], [data_val], nb_epoch=2,
                        batch_size=batch_size, metrics=metrics,
                        custom_objects=cust_objects)
                print(excinfo)
        else:
            expe.fit([data], [data_val], nb_epoch=2,
                    batch_size=batch_size, metrics=metrics,
                    custom_objects=cust_objects)

            expe.fit([data], [data_val], model=model, nb_epoch=2,
                    batch_size=batch_size, metrics=metrics,
                    custom_objects=cust_objects)

            expe.backend_name = 'another_backend'
            expe.load_model()
            expe.load_model(expe.mod_id, expe.data_id)

            assert expe.data_id is not None
            assert expe.mod_id is not None
            assert expe.params_dump is not None
        print(self)

    def test_experiment_fit_async(self, get_model, get_loss_metric,
                                  get_custom_l):
        data, data_val = make_data()
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        expe = Experiment(model)

        if get_custom_l:
            with pytest.raises(RuntimeError) as excinfo:
                res, th = expe.fit_async([data], [data_val], nb_epoch=2,
                                         batch_size=batch_size,
                                         metrics=metrics,
                                         custom_objects=cust_objects)
                print(excinfo)
                # th.join()
        else:
            res, th = expe.fit_async([data], [data_val], nb_epoch=2,
                                     batch_size=batch_size, metrics=metrics,
                                     custom_objects=cust_objects)

            # th.join()
            res, th = expe.fit_async([data], [data_val], model=model,
                                     nb_epoch=2,
                                     batch_size=batch_size,
                                     metrics=metrics,
                                     custom_objects=cust_objects)
            # th.join()
        print(self)

    def test_experiment_fit_gen(self, get_model, get_loss_metric,
                                get_custom_l):
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        model_name = model.__class__.__name__
        is_graph = model_name.lower() == 'graph'
        _, data_val_use = make_data()
        expe = Experiment(model)
        gen, data, data_stream = make_gen(is_graph)


        if get_custom_l:
            for val in [gen, data_val_use]:
                with pytest.raises(RuntimeError) as excinfo:
                    expe.fit_gen([gen], [val], nb_epoch=2,
                                 model=model,
                                 metrics=metrics,
                                 custom_objects=cust_objects,
                                 samples_per_epoch=128,
                                 nb_val_samples=128)
                    print(excinfo)
        else:
            for val in [gen, data_val_use]:
                gen, data, data_stream = make_gen(is_graph)
                expe.fit_gen([gen], [val], nb_epoch=2,
                                model=model,
                                metrics=metrics,
                                custom_objects=cust_objects,
                                samples_per_epoch=128,
                                nb_val_samples=128)
                gen.close()
                data.close(None)
                data_stream.close()

            assert expe.data_id is not None
            assert expe.mod_id is not None
            assert expe.params_dump is not None

        gen.close()
        data.close(None)
        data_stream.close()
        print(self)

    def test_experiment_fit_gen_async(self, get_model, get_loss_metric,
                                      get_custom_l):
        model, metrics, cust_objects = prepare_model(get_model(get_custom_l),
                                                     get_loss_metric,
                                                     get_custom_l)

        model_name = model.__class__.__name__
        is_graph = model_name.lower() == 'graph'
        _, data_val_use = make_data()
        expe = Experiment(model)
        gen, data, data_stream = make_gen(is_graph)

        if get_custom_l:
            for val in [gen, data_val_use]:
                with pytest.raises(RuntimeError) as excinfo:
                    res, th = expe.fit_gen_async([gen], [val], nb_epoch=2,
                                                 model=model,
                                                 metrics=metrics,
                                                 custom_objects=cust_objects,
                                                 samples_per_epoch=128,
                                                 nb_val_samples=128)
                    print(excinfo)
                    # th.join()
        else:
            for val in [gen, data_val_use]:
                res, th = expe.fit_gen_async([gen], [val], nb_epoch=2,
                                                 model=model,
                                                 metrics=metrics,
                                                 custom_objects=cust_objects,
                                                 samples_per_epoch=128,
                                                 nb_val_samples=128)
                # th.join()

        gen.close()
        data.close(None)
        data_stream.close()
        print(self)

    def test_experiment_predict(self, get_model, get_loss_metric):

        model, metrics, cust_objects = prepare_model(get_model(),
                                                     get_loss_metric,
                                                     False)

        model_name = model.__class__.__name__
        data, data_val = make_data()

        expe = Experiment(model)
        expe.fit([data], [data_val], nb_epoch=2,
                 batch_size=batch_size,
                 custom_objects=cust_objects,
                 metrics=metrics)

        if model_name == 'Graph' or model_name == 'Model':
            expe.predict({'X': data_val['X']})
        expe.predict([data_val['X']])
        expe.predict(data_val['X'])
        print(self)


class TestBackendFunctions:
    def test_build_predict_func(self, get_model):
        """Test the build of a model"""
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
        print(self)

    def test_fit(self, get_model):
        "Test the training of a serialized model"
        data, data_val = make_data()

        model = get_model()
        model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        model_dict = dict()
        model_dict['model_arch'] = to_dict_w_opt(model)

        res = KTB.train(copy.deepcopy(model_dict['model_arch']), [data], [data_val])
        res = KTB.fit(NAME, VERSION, model_dict, [data], 'test', [data_val])

        assert len(res) == 4
        print(self)

    def test_predict(self, get_model):
        """Test to predict using the backend"""
        data, data_val = make_data()
        model = get_model()
        model.compile(optimizer='sgd', loss='categorical_crossentropy')

        expe = Experiment(model)
        expe.fit([data], [data_val])
        KTB.predict(expe.model_dict, [data['X']])
        print(self)


def test_utils():
    assert get_function_name("bob") == "bob"
    test_switch = switch_backend('sklearn')
    assert test_switch is not None
    gen, data, data_stream = make_gen()
    open_dataset_gen(data_stream)


if __name__ == "__main__":
    pytest.main([__file__])
