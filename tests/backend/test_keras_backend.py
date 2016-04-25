"""Tests for the Keras backend"""

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
from alp.backend import keras_backend as KTB
from alp.backend.utils.keras_utils import get_function_name
from alp.backend.utils.keras_utils import to_dict_w_opt

np.random.seed(1337)


input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 20
test_samples = 20


def test_build_predict_func():
    """Test the build of a model"""
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                                nb_test=test_samples,
                                                input_shape=(input_dim,),
                                                classification=True,
                                                nb_class=nb_class)

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
    res = pred_func([X_te])

    assert len(res[0]) == len(X_te)


def test_fit():
    "Test the training of a serialized model"
    import keras.backend as K
    def categorical_crossentropy(y_true, y_pred):
        '''A test of custom loss function
        '''
        return K.categorical_crossentropy(y_pred, y_true)

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

    custom_objects = {"categorical_crossentropy": categorical_crossentropy}

    # Case 1 sequential model
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_json = to_dict_w_opt(model)

    res = KTB.fit(model_json, [data], [data_val])

    assert len(res) == 2

    # Case 3 without custom objects
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_json = to_dict_w_opt(model)

    res = KTB.fit(model_json, [data], [data_val])

    assert len(res) == 2

    data, data_val = dict(), dict()

    data["X_vars"] = X_tr
    data["output"] = y_tr

    data_val["X_vars"] = X_te
    data_val["output"] = y_te

    # Case 2 Graph model
    model = Graph()
    model.add_input(name='X_vars', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X_vars')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')

    model.add_output(name='output', input='last_dense')
    model.compile(optimizer='sgd', loss={'output': categorical_crossentropy})

    model_json = to_dict_w_opt(model)
    res = KTB.fit(model_json, [data], [data_val])

    assert len(res) == 2

    model_json = to_dict_w_opt(model)
    res = KTB.fit(model_json, [data], [data_val])
                          
    assert len(res) == 2


def test_utils():
    assert get_function_name("bob") == "bob"


def test_experiment():
    """Test the Experiment class"""
    import keras.backend as K

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


    def categorical_crossentropy(y_true, y_pred):
        '''A test of custom loss function
        '''
        return K.categorical_crossentropy(y_pred, y_true)

    custom_objects = {"categorical_crossentropy": categorical_crossentropy}

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    expe = Experiment("keras", model)

    assert expe.backend is not None

    expe.fit([data], [data_val], custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)

    expe.fit([data], [data_val], model=model,
             custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)
    expe.predict(data)

    # Case 2 Graph model

    data = dict()
    data_val = dict()

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
    model.compile(optimizer='sgd', loss={'output': categorical_crossentropy})

    expe = Experiment("keras", model)

    assert expe.backend is not None

    expe.fit([data], [data_val], custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)
    expe.fit([data], [data_val], model=model,
             custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)
    expe.predict(data)
    expe.trained = False
    expe.predict(data)

    # Case 3 Model
    # this returns a tensor
    inputs = Input(shape=(input_dim,))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(nb_hidden, activation='relu')(inputs)
    x = Dense(nb_hidden, activation='relu')(x)
    predictions = Dense(nb_class, activation='softmax')(x)

    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(input=inputs, output=predictions)
    model.compile(optimizer='rmsprop',
                loss=categorical_crossentropy,
                metrics=['accuracy'])


if __name__ == "__main__":
    pytest.main([__file__])
