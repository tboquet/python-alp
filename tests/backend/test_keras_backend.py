"""Tests for the Keras backend"""
import pytest
import os
import sys
import numpy as np
np.random.seed(1337)

from keras import callbacks
from keras.models import Graph, Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import get_test_data
from keras import backend as K
from keras.utils import np_utils


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

    pred_func = build_predict_func(model)
    res = pred_func([X_test])

    assert 0 == 0


def test_train_model():
    "Test the training of a serialized model"
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                               nb_test=test_samples,
                                               input_shape=(input_dim,),
                                               classification=True,
                                               nb_class=nb_class)

    y_train = np_utils.to_categorical(y_tr)

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

    model_str = model.to_json()
    res = train_model(model_str, [], datas, datas_val, batch_size,
                      2, [])

    assert 0 == 0
