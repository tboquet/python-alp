"""Tests for the Keras backend"""

import keras
import keras.backend as K
import numpy as np
import pytest

from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift
from keras.engine import Layer
from keras.layers import Dense
from keras.layers import Input
from keras.models import Graph
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

from alp.appcom.core import Experiment
from alp.appcom.utils import switch_backend
from alp.appcom.utils import to_fuel_h5
from alp.backend import keras_backend as KTB
from alp.backend.keras_backend import get_function_name
from alp.backend.keras_backend import to_dict_w_opt

np.random.seed(1337)


input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 128
test_samples = 64
NAME = keras.__name__
VERSION = keras.__version__


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


def _test_experiment(model, custom_objects=None):
    from alp.appcom.utils import imports

    if custom_objects is None:
        custom_objects = dict()
    @imports({"K": K})
    def categorical_crossentropy_custom(y_true, y_pred):
        '''A test of custom loss function
        '''
        return K.categorical_crossentropy(y_pred, y_true)

    @imports({"K": K})
    def cosine_proximity(y_true, y_pred):
        y_true = K.l2_normalize(y_true, axis=-1)
        y_pred = K.l2_normalize(y_pred, axis=-1)
        return -K.mean(y_true * y_pred)

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

    metrics = ['accuracy', cosine_proximity]

    cust_objects = dict()
    cust_objects['categorical_crossentropy_custom'] = categorical_crossentropy_custom
    cust_objects.update(custom_objects)

    model.compile(loss=categorical_crossentropy_custom,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    expe = Experiment(model)

    assert expe.backend is not None

    expe.fit([data], [data_val], custom_objects=cust_objects, nb_epoch=2,
             batch_size=batch_size, metrics=metrics)

    # check data_id
    assert expe.data_id is not None

    # check mod_id
    assert expe.mod_id is not None

    # check params_dump
    assert expe.params_dump is not None

    # async
    expe.fit_async([data], [data_val], custom_objects=cust_objects,
                   nb_epoch=2, batch_size=batch_size)

    # try to reload the same model
    expe.backend_name = "test"
    expe.load_model()

    expe.load_model(expe.mod_id, expe.data_id)
    # check the serialization of the model
    expe.model_dict = model

    expe.fit([data], [data_val], model=model,
             custom_objects=cust_objects, nb_epoch=2,
             batch_size=batch_size)
    expe.predict(data['X'].astype('float32'))

    # check if the cached model is used
    expe.predict(data['X'].astype('float32'))
    expe.predict([data['X'].astype('float32')])
    if model.__class__.__name__ != 'Sequential':
        expe.predict({k: data[k].astype('float32') for k in data})

    model.compile(loss=[categorical_crossentropy_custom],
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    expe = Experiment(model)

    expe.fit([data], [data_val], model=model,
             custom_objects=cust_objects, nb_epoch=2,
             batch_size=batch_size)

    expe.fit_async([data], [data_val], model=model, metrics=metrics,
                   custom_objects=cust_objects,
                   nb_epoch=2, batch_size=batch_size)

    inputs = [np.concatenate([data['X'], data_val['X']])]
    outputs = [np.concatenate([data['y'], data_val['y']])]

    scale = 1.0 / inputs[0].std(axis=0)
    shift = - scale * inputs[0].mean(axis=0)

    model_name = model.__class__.__name__

    if model_name  == 'Graph':
        inp_name = model.input_names[0]
        out_name = model.output_names[0]
        inputs = dict()
        outputs = dict()
        inputs[inp_name] = np.concatenate([data['X'], data_val['X']])
        outputs[out_name] = np.concatenate([data['y'], data_val['y']])

    full_path = to_fuel_h5(inputs, outputs, [0, 164], ['train', 'test'],
                           'test_data' + str(model_name))

    train_set = H5PYDataset(full_path, which_sets=('train','test'))

    state_train = train_set.open()

    scheme = SequentialScheme(examples=128, batch_size=32)

    data_stream_train = DataStream(dataset=train_set, iteration_scheme=scheme)

    stand_stream_train = ScaleAndShift(data_stream=data_stream_train,
                                        scale=scale, shift=shift,
                                        which_sources=('input_X',))

    print("Gen simple generator")
    expe.fit_gen([stand_stream_train], [data_val],
                 model=model,
                 metrics=metrics,
                 custom_objects=cust_objects,
                 nb_epoch=2,
                 samples_per_epoch=128)

    print("Gen double generator")
    expe.fit_gen([stand_stream_train], [stand_stream_train],
                 model=model,
                 metrics=metrics,
                 custom_objects=cust_objects,
                 nb_epoch=2,
                 samples_per_epoch=128,
                 nb_val_samples=128)

    print("Gen async")
    expe.fit_gen_async([stand_stream_train], [stand_stream_train],
                       model=model,
                       metrics=metrics,
                       custom_objects=cust_objects,
                       nb_epoch=2,
                       samples_per_epoch=128,
                       nb_val_samples=128)

    stand_stream_train.close()
    data_stream_train.close()
    train_set.close(state_train)


def test_build_predict_func():
    """Test the build of a model"""
    X_tr = np.ones((train_samples, input_dim))
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    pred_func = KTB.build_predict_func(model)
    res = pred_func([X_tr])

    assert len(res[0]) == len(X_tr)

    model = Graph()
    model.add_input(name='X_vars', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X_vars')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')
    model.add_output(name='output', input='last_dense')
    model.compile(optimizer='sgd', loss={'output': 'mse'})

    pred_func = KTB.build_predict_func(model)
    res = pred_func([X_tr])

    assert len(res[0]) == len(X_tr)


def test_fit():
    "Test the training of a serialized model"

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

    # Case 1 sequential model
    metrics = ['accuracy']

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=metrics)

    model_dict = dict()
    model_dict['model_arch'] = to_dict_w_opt(model, metrics)

    res = KTB.train(model_dict['model_arch'], [data], [data_val])
    res = KTB.fit(NAME, VERSION, model_dict, [data], 'test', [data_val])

    assert len(res) == 4
    # Case 2 without custom objects
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_dict = dict()
    model_dict['model_arch'] = to_dict_w_opt(model, metrics)

    res = KTB.fit(NAME, VERSION, model_dict, [data], 'test', [data_val])

    assert len(res) == 4
    # Case 3 Graph model

    data, data_val = dict(), dict()

    data["X_vars"] = X_tr
    data["output"] = y_tr

    data_val["X_vars"] = X_te
    data_val["output"] = y_te

    model = Graph()
    model.add_input(name='X_vars', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X_vars')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')

    model.add_output(name='output', input='last_dense')
    model.compile(optimizer='sgd', loss={'output': 'categorical_crossentropy'})

    model_dict = dict()
    model_dict['model_arch'] = to_dict_w_opt(model, metrics)

    res = KTB.fit(NAME, VERSION, model_dict, [data], 'test', [data_val])

    assert len(res) == 4


def test_predict():
    """Test the prediction using the backend"""
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                                nb_test=test_samples,
                                                input_shape=(input_dim,),
                                                classification=True,
                                                nb_class=nb_class)

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)

    data, data_val = dict(), dict()

    data["X_vars"] = X_tr
    data["output"] = y_tr

    data_val["X_vars"] = X_te
    data_val["output"] = y_te
    model = Graph()
    model.add_input(name='X_vars', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X_vars')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')

    model.add_output(name='output', input='last_dense')
    model.compile(optimizer='sgd', loss={'output': 'categorical_crossentropy'})

    expe = Experiment(model)
    expe.fit([data], [data_val])
    KTB.predict(expe.model_dict, [data['X_vars']])


def test_utils():
    assert get_function_name("bob") == "bob"
    test_switch = switch_backend('sklearn')
    assert test_switch is not None


def test_experiment_sequential():
    """Test the Experiment class with Sequential"""
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))

    custom_objects = {'Dropout_cust': Dropout_cust}
    _test_experiment(model)


def test_experiment_model():
    """Test the Experiment class with Model"""


    inputs = Input(shape=(input_dim,), name='X')

    x = Dense(nb_hidden, activation='relu')(inputs)
    x = Dense(nb_hidden, activation='relu')(x)
    predictions = Dense(nb_class, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)
    _test_experiment(model)


def test_experiment_legacy():
    """Test the Experiment class with Model"""
    model = Graph()
    model.add_input(name='X', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')

    model.add_output(name='y', input='last_dense')
    _test_experiment(model)


if __name__ == "__main__":
    pytest.main([__file__])
