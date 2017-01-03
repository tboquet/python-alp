import numpy as np
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import ScaleAndShift
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.test_utils import get_test_data

from alp.appcom.utils import to_fuel_h5


input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 4
train_samples = 256
test_samples = 128


def close_gens(gen, data, data_stream):
    gen.close()
    data.close(None)
    data_stream.close()


def make_data(train_samples, test_samples):
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


def dump_data(train_samples, test_samples):
    data, data_val = make_data(train_samples, test_samples)
    inputs = [np.concatenate([data['X'], data_val['X']])]
    outputs = [np.concatenate([data['y'], data_val['y']])]

    file_name = 'test_data'
    scale = 1.0 / inputs[0].std(axis=0)
    shift = - scale * inputs[0].mean(axis=0)

    file_path, i_names, o_names = to_fuel_h5(inputs, outputs, [0, 256],
                                             ['train', 'test'],
                                             file_name,
                                             '/data_generator')
    return file_path, scale, shift, i_names, o_names


file_path, scale, shift, i_names, o_names = dump_data(train_samples, test_samples)


def make_gen(batch_size, examples=4):
    file_path_f = file_path
    names_select = i_names
    train_set = H5PYDataset(file_path_f,
                            which_sets=('train', 'test'))

    scheme = SequentialScheme(examples=examples, batch_size=batch_size)

    data_stream_train = DataStream(dataset=train_set, iteration_scheme=scheme)

    stand_stream_train = ScaleAndShift(data_stream=data_stream_train,
                                       scale=scale, shift=shift,
                                       which_sources=(names_select[-1],))
    return stand_stream_train, train_set, data_stream_train


def sequential(custom=False, nb_hidden=4):
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.add(Dropout(0.5))
    if custom:
        model.add(return_custom()(0.5))
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


def return_custom():
    import keras.backend as K
    from keras.engine import Layer

    class Dropout_cust(Layer):  # pragma: no cover
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

    return Dropout_cust
