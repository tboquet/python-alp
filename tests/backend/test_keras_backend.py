"""Tests for the Keras backend"""

import keras
import numpy as np
import pytest
from keras.layers import Dense
from keras.layers import Input
from keras.models import Graph
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

from alp.appcom.core import Experiment
from alp.appcom.utils import switch_backend
from alp.backend import keras_backend as KTB
from alp.backend.keras_backend import get_function_name
from alp.backend.keras_backend import to_dict_w_opt

np.random.seed(1337)


input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 20
test_samples = 20
NAME = keras.__name__
VERSION = keras.__version__


def _test_experiment(model):
    from alp.appcom.utils import imports
    import keras.backend as K

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

    custom_objects = dict()
    custom_objects['categorical_crossentropy_custom'] = categorical_crossentropy_custom

    model.compile(loss=categorical_crossentropy_custom,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    expe = Experiment(model)

    assert expe.backend is not None

    expe.fit([data], [data_val], custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size, metrics=metrics)

    # check data_id
    assert expe.data_id is not None

    # check mod_id
    assert expe.mod_id is not None

    # check params_dump
    assert expe.params_dump is not None

    # async
    expe.fit_async([data], [data_val], custom_objects=custom_objects,
                   nb_epoch=2, batch_size=batch_size)

    # try to reload the same model
    expe.backend_name = "test"
    expe.load_model(expe.mod_id, expe.data_id)

    # check the serialization of the model
    expe.model_dict = model

    expe.fit([data], [data_val], model=model,
             custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)
    expe.predict(data['X'].astype('float32'))

    # check if the cached model is used
    expe.predict(data['X'].astype('float32'))
    expe.predict([data['X'].astype('float32')])
    if model.__class__.__name__ == 'Graph':
        expe.predict({k: data[k].astype('float32') for k in data})

    model.compile(loss=[categorical_crossentropy_custom],
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    expe = Experiment(model)

    expe.fit([data], [data_val], model=model,
             custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)

    expe.fit_async([data], [data_val], model=model, metrics=metrics,
                   custom_objects=custom_objects,
                   nb_epoch=2, batch_size=batch_size)


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
    res = KTB.fit(NAME, VERSION, model_dict, [data], [data_val])

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

    res = KTB.fit(NAME, VERSION, model_dict, [data], [data_val])

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

    res = KTB.fit(NAME, VERSION, model_dict, [data], [data_val])

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
    res = expe.fit([data], [data_val])
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
