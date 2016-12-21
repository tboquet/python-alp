========================================================
Tutorial 4 : how to use custom layers for Keras with ALP
========================================================

Because serialization of complex Python objects is still a challenge we will present a way of sending a custom layer to a Keras model with ALP.


We will work with the CIFAR10 dataset available via Keras.

.. code-block:: python

    from keras.datasets import cifar10
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD
    from keras.utils import np_utils

    from fuel.datasets.hdf5 import H5PYDataset
    from fuel.schemes import SequentialScheme
    from fuel.streams import DataStream
    from fuel.transformers import ScaleAndShift

    from alp.appcom.core import Experiment

    from alp.appcom.utils import to_fuel_h5

    import numpy as np

    nb_classes = 10
    nb_epoch = 25

    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train/255
    X_test = X_test/255

    batch_size = 128
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)


We build two generators, one for training and one for validation.


.. code-block:: python

    def dump_data():
        inputs = [np.concatenate([X_train, X_test])]
        outputs = [np.concatenate([Y_train, Y_test])]

        file_name = 'test_data_dropout'
        scale = 1.0 / inputs[0].std(axis=0)
        shift = - scale * inputs[0].mean(axis=0)

        file_path, i_names, o_names = to_fuel_h5(inputs, outputs, [0, 50000],
                                                ['train', 'test'],
                                                file_name,
                                                '/data_generator')
        return file_path, scale, shift, i_names, o_names

    file_path, scale, shift, i_names, o_names = dump_data()


    def make_gen(set_to_gen, nb_examples):
        file_path_f = file_path
        names_select = i_names
        train_set = H5PYDataset(file_path_f,
                                which_sets=set_to_gen)

        scheme = SequentialScheme(examples=nb_examples, batch_size=64)

        data_stream_train = DataStream(dataset=train_set, iteration_scheme=scheme)

        stand_stream_train = ScaleAndShift(data_stream=data_stream_train,
                                          scale=scale, shift=shift,
                                          which_sources=(names_select[-1],))
        return stand_stream_train, train_set, data_stream_train

    train, data_tr, data_stream_tr = make_gen(('train',), 50000)
    test, data_te, data_stream_te = make_gen(('test',), 10000)

Imagine you want to reimplement a dropout layer. We could wrap it in a function that returns the object:


.. code-block:: python

  def return_custom():
      import keras.backend as K
      import numpy as np
      from keras.engine import Layer
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
      return Dropout_cust


We then define our model and call our function to instanciate this custom layer.

.. code-block:: python

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(return_custom()(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.02, decay=1e-7, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])


We then map the name of the custom object to our function that returns the custom object in a dictionnary.
After wrapping the model in an :meth:`alp.appcom.core.Experiment`, we call the :meth:`alp.appcom.core.Experiment.fit_gen` method and send the custom_objects.

.. code-block:: python

    custom_objects = {'Dropout_p': return_custom}

    expe = Experiment(model)

    results = expe.fit_gen([train], [test], nb_epoch=nb_epoch,
                           model=model,
                           metrics=['accuracy'],
                           samples_per_epoch=50000,
                           nb_val_samples=10000,
                           verbose=2,
                           custom_objects=custom_objects))


Why do we wrap this class and all the dependencies?

We use dill to be able to serialize object but unfortunatly, handling class with inheritance is not doable. It's also easier to pass the information about all the dependencies of the object. All the dependencies and your custom objects will be instanciated during the evaluation of the function so that it will be available in the `__main__`. This way the information could be sent to workers without problems.
