========
Overview
========

.. start-badges

|travis| |requires| |coveralls| |codecov| |codacy| |docs|

.. |travis| image:: https://travis-ci.org/tboquet/python-alp.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/tboquet/python-alp

.. |requires| image:: https://requires.io/github/tboquet/python-alp/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/tboquet/python-alp/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/tboquet/python-alp/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/tboquet/python-alp

.. |codecov| image:: https://codecov.io/github/tboquet/python-alp/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/tboquet/python-alp

.. |codacy| image:: https://img.shields.io/codacy/b7f6d79244d8480099a3593db2de9560.svg?style=flat
    :target: https://www.codacy.com/app/tboquet/python-alp
    :alt: Codacy Code Quality Status

.. |docs| image:: https://readthedocs.org/projects/python-alp/badge/?style=flat
    :target: https://readthedocs.org/projects/python-alp
    :alt: Documentation Status

.. end-badges


ALP helps you experiment with a lot of machine learning models quickly. It provides you with a simple way of scheduling and recording experiments.

This library has been developped to work well with Keras and Scikit-learn but can suit a lot of other frameworks. 

Documentation
=============

http://python-alp.readthedocs.io/

Quickstart
==========

Clone the repo and install the library:

.. code-block:: bash

   git clone https://github.com/tboquet/python-alp.git
   cd python-alp
   python setup.py install

Install the Command Line Interface dependencies:

.. code-block:: bash

   cd req
   pip install requirements_cli.txt

Generate a base configuration:

.. code-block:: bash

    alp --verbose genconfig --outdir=/path/to/a/directory --cpu

Launch the services:

.. code-block:: bash

    alp --verbose service start /path/to/a/directory

Check the status of your containers:

.. code-block:: bash

    alp --verbose status /path/to/a/directory


Log in to the Jupyter notebook you just launched in your browser @ :code:`localhost:440` using the password :code:`default`.

Launch some experiments!

.. code-block:: python

    # we import numpy and fix the seed
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    # we import alp and Keras tools that we will use
    import alp
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.utils import np_utils
    import keras.backend as K
    from keras.optimizers import Adam
    from alp.appcom.ensembles import HParamsSearch

    # if you use tensorflow you must use this configuration
    # so that it doesn't use all of the GPU's memory (default config)
    import tensorflow as tf

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)

    batch_size = 128
    nb_classes = 10
    nb_epoch = 12

    # input image dimensions
    img_rows, img_cols = 28, 28
    # number of features to use
    nb_features = 32

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # put the data in the form ALP expects
    data, data_val = dict(), dict()
    data["X"] = X_train[:500]
    data["y"] = Y_train[:500]
    data_val["X"] = X_test[:500]
    data_val["y"] = Y_test[:500]

    # Define and compile the model

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(nb_features))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Define you experiment

    from alp.appcom.core import Experiment

    expe = Experiment(model)

    # Fit the model linked to your experiment
    results = expe.fit([data], [data_val], nb_epoch=2, batch_size=batch_size)

    # Predict using your model
    expe.predict(data['X'])


`Get started with the tutorial series!`_ 

* Free software: Apache license

.. _`docker setup`: http://python-alp.readthedocs.io/en/latest/dockersetup.html
.. _`Get started with the tutorial series!`: http://python-alp.readthedocs.io/en/latest/Tutorials/index_tuto.html
