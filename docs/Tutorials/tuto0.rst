===================================================================
Tutorial 0 : how to launch a basic experiment with keras or sklearn
===================================================================

Step 1 : launching alp
~~~~~~~~~~~~~~~~~~~~~~


Follow the instructions in the setup section.
We assume at this point that you have a Jupyter notebook running on the controller.


Step 2 : defining your model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can follow step from `Step 2.1 : Keras`_ or from `Step 2.2 : Scikit learn`_ regarding if you want to use Keras_ or `scikit-learn`_. In both case we will do the right imports, get some classification data, put them in the ALP format and instanciate a model. The important thing at the end of step 2 is to have the :code:`data`, :code:`data_val` and :code:`model` objects and a model ready.

Step 2.1 : Keras
++++++++++++++++

The following code gets some data and declares a simple artificial neural network with Keras:

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
    nb_filters = 32

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
    data["X"] = X_train
    data["y"] = Y_train
    data_val["X"] = X_test
    data_val["y"] = Y_test

    # finally define and compile the model

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(nb_filters))
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

Note that we compile the model so that we also have information about the optimizer.


Step 2.2 : Scikit learn
+++++++++++++++++++++++

The following code gets some data and declares a simple logistic regression with :code:`scikit-learn`:

.. code-block:: python
    
    # some imports
    from sklearn import cross_validation
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    
    # get some data
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                    iris.data, iris.target, test_size=0.2, random_state=0)

    # put the data in the form ALP expects
    data, data_val = dict(), dict()
    data["X"] = X_train
    data["y"] = y_train
    data_val["X"] = X_test
    data_val["y"] = y_test
   
    # define the model
    model = LogisticRegression()

Please note that by default for the :code:`LogisticRegression`, the :code:`multi-class` parameter is set to OvR, that is to say one classifier per class. On the iris dataset, it means 3 classifiers. Unlike in Keras, the model is not compiled. So far, the measure of performance (validation metric) can only be the mean absolute error, but we will soon have several metrics working.


Step 3 : fitting the model with ALP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 3.1 : defining the Experiment
++++++++++++++++++++++++++++++++++

In ALP, the base object is the Experiment.
An Experiment trains, predicts, saves and logs a model.
So the first step is to import and define the Experiment object.

.. code-block:: python

    from alp.appcom.core import Experiment

    expe = Experiment(model)


Step 3.2 : fit the model 
++++++++++++++++++++++++

You have access to two types of methods to fit the model.

* The :code:`fit` and :code:`fit_gen` methods allows you to fit the model in the same process.

	For the :code:`scikit-learn` backend, you can launch the computation with the following command without extra arguments:

	.. code-block:: python

	    expe.fit([data], [data_val])

	Note that the :code:`data` and the :code:`data_val` are put in lists.


	With Keras you might want to specify the number of epochs and the batch_size, as you would have done to fit directly a Keras :code:`model` object. These arguments will flow trough to the final call. Note that they are not necessary for the fit, see the default arguments in the `Keras model doc <https://keras.io/models/model/>`_.

	.. code-block:: python

	    expe.fit([data], [data_val], nb_epoch=2, batch_size=batch_size)

	In both cases, the model is trained and automatically saved in the databases.

* The :code:`fit_async` method sends the model to the broker container that will manage the training using the workers you defined in the setup phase. The commands are then straightforward:
	For the :code:`scikit-learn` backend:

	.. code-block:: python

	    expe.fit_async([data], [data_val])


	For the Keras backend you still need to provide extra arguments to override the defaults.

	.. code-block:: python

	    expe.fit_async([data], [data_val], nb_epoch=2, batch_size=batch_size)

	In both cases, the model is also trained and automatically saved in the databases.



Step 4 : Identifying and reusing the fitted model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the experiment has been fitted, you can access the id of the model in the db and load it to make prediction or access the parameters in the current process.

.. code-block:: python

    print(expe.mod_id)
    print(expe.data_id)

    expe.load_model(expe.mod_id, expe.data_id)


It's then possible to make predictions using the loaded model.

.. code-block:: python

    expe.predict(data['X'])

You could of course provide new data to the model. You can also load the model in another experiment.

.. _Keras: http://keras.io/
.. _`scikit-learn`: http://scikit-learn.org/stable/
