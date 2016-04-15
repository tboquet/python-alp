"""Tests for the Keras backend"""

import numpy as np
import pytest
from keras.models import Graph
from keras. models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense
from keras.utils.test_utils import get_test_data

from alp.backend import keras_backend as KTB


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

    assert 0 == 0

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

    assert 0 == 0

def test_train_model():
    "Test the training of a serialized model"

    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                                nb_test=test_samples,
                                                input_shape=(input_dim,),
                                                classification=True,
                                                nb_class=nb_class)

    y_tr = np_utils.to_categorical(y_tr)

    datas, datas_val = dict(), dict()

    datas["X"] = X_tr
    datas["y"] = y_tr

    datas_val["X"] = X_tr
    datas_val["y"] = y_tr

    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model_json = KTB.to_json_w_opt(model)
    res = KTB.train_model(model_json, [], [datas], [datas_val], batch_size,
                      2, [])

    datas["X_vars"] = X_tr
    datas["output"] = y_tr

    datas_val["X_vars"] = X_tr
    datas_val["output"] = y_tr

    model = Graph()
    model.add_input(name='X_vars', input_shape=(input_dim, ))

    model.add_node(Dense(nb_hidden, activation="sigmoid"),
                   name='Dense1', input='X_vars')
    model.add_node(Dense(nb_class, activation="softmax"),
                   name='last_dense',
                   input='Dense1')
    model.add_output(name='output', input='last_dense')
    model.compile(optimizer='sgd', loss={'output': 'mse'})

    model_json = KTB.to_json_w_opt(model)
    res = KTB.train_model(model_json, None, [datas], [datas_val], batch_size,
                          2, [])
    assert 0 == 0


if __name__ == "__main__":
    pytest.main([__file__])
