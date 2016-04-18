"""Tests for the Keras backend"""

import numpy as np
import pytest
from keras.layers.core import Dense
from keras.models import Graph
from keras. models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

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

    y_tr -= 0
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    pred_func = KTB.build_predict_func(model)
    res = pred_func([X_te])


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


def test_train_model():
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

    datas, datas_val = dict(), dict()

    datas["X"] = X_tr
    datas["y"] = y_tr

    datas_val["X"] = X_te
    datas_val["y"] = y_te

    custom_objects = {"categorical_crossentropy": categorical_crossentropy}

    # Case 1 sequential model
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_json = to_dict_w_opt(model)
    res = KTB.train_model(model_json, [datas], [datas_val], batch_size,
                          2, None, custom_objects)

    assert length(res[0]) == 2

    # Case 3 without custom objects
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_json = to_dict_w_opt(model)
    res = KTB.train_model(model_json, [datas], [datas_val], batch_size,
                          2, [])

    assert length(res[0]) == 2

    datas["X_vars"] = X_tr
    datas["output"] = y_tr

    datas_val["X_vars"] = X_te
    datas_val["output"] = y_te

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
    res = KTB.train_model(model_json, [datas], [datas_val], batch_size,
                          2, [], custom_objects)

    assert length(res[0]) == 2


def test_utils():
    assert get_function_name("bob") == "bob"


if __name__ == "__main__":
    pytest.main([__file__])
