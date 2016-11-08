=========================
Getting started with ALP!
=========================

Please go through the environment setup (see :ref:`Requirements`, and :ref:`Set up of your containers`) before going further. You should have your docker containers ready to accept jobs.
After this initial setup, we are all set to start experimenting with Keras (and some models of scikit-learn)!

For now ALP fully supports Keras_ and partially supports `scikit-learn`_ (linear models).

Keras example
-------------

We will begin by declaring a simple artificial neural network with Keras_:

.. code-block:: python

    from keras.layers import Dense
    from keras.layers import Input
    from keras.models import Sequential
    from keras.utils import np_utils
    from keras.utils.test_utils import get_test_data

    input_dim = 2
    nb_hidden = 4
    nb_class = 2
    batch_size = 5
    train_samples = 20
    test_samples = 20

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

    metrics = ['accuracy']

    custom_objects = dict()
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


Note that we compile the model so that we also have information about the optimizer.


Fitting the model using ALP
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We then instanciate an :class:`alp.appcom.core.Experiment`:

.. code-block:: python

    from alp.appcom.core import Experiment

    expe = Experiment(model)
    

Then, you have access to two methods to fit the model.
The :meth:`alp.appcom.core.Experiment.fit` method allows you to fit the model in the same process.

.. code-block:: python

    expe.fit([data], [data_val], custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)


Here, you will see the regular print output of Keras. The model is being trained and automatically saved in the database. 


The :meth:`alp.appcom.core.Experiment.fit_async` method send the model to the broker container that will manage the training using the workers you defined in the setup phase.

.. code-block:: python

    expe.fit_async([data], [data_val], custom_objects=custom_objects,
                   nb_epoch=2, batch_size=batch_size)


For now, we don't directly redirect the training information from the worker to a web application or a log so you can have it in real time. This feature is on the todo list and will be implemented in the following weeks.
Like for the fit method, the architecture of the model is saved in the db along with the performance and the parameters are dumped in an HDF5 file.


Predictions using the model saved in the database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the experiment has been fitted, you can access the id of the model in the db and load it to make prediction or access the parameters in the current process.

.. code-block:: python

    print(expe.mod_id)
    print(expe.data_id)

    expe.load_model(expe.mod_id, expe.data_id)


It's then possible to make predictions using the loaded model.

.. code-block:: python

    expe.predict(data['X'].astype('float32'))



Sklearn example
---------------

As previously said, another partially supported backend is `scikit-learn`_. Let's do some logistic regression.

First we need to get some data. The iris dataset is what we want.

.. code-block:: python

    from sklearn import cross_validation
    from sklearn import datasets
    from sklearn.linear_model import LogisticRegression
    
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                    iris.data, iris.target, test_size=0.2, random_state=0)

    data, data_val = dict(), dict()

    data["X"] = X_train
    data["y"] = y_train

    data_val["X"] = X_test
    data_val["y"] = y_test

Definining a model in scikit-learn is simple. Here we will create a basic instance of a LogisticRegression. Please note that by default, the 'multi-class' parameter is set to OvR, that is to say one classifier per class. On the iris dataset, it means 3 classifiers.

.. code-block:: python

    lr = LogisticRegression()

Unlike in Keras, the model is not compiled.
A word on performance : so far, the measure of performance is the mean absolute error, but we will soon have several metrics working.

Fitting the model using ALP
~~~~~~~~~~~~~~~~~~~~~~~~~~~


We then instanciate an :class:`alp.appcom.core.Experiment`:

.. code-block:: python

    from alp.appcom.core import Experiment

    expe = Experiment(lr)
 
The two methods (direct and asynchronous) are available, since we just switched backend.
The :meth:`alp.appcom.core.Experiment.fit` method allows you to fit the model in the same process.

.. code-block:: python

    expe.fit([data], [data_val])

The model is being trained and automatically saved in the database. 

The :meth:`alp.appcom.core.Experiment.fit_async` method send the model to the broker container that will manage the training using the workers you defined in the setup phase.

.. code-block:: python

    expe.fit_async([data], [data_val])

Like for the fit method, the model is saved in the db along with the performance and the parameters are dumped in an HDF5 file.  



Predictions using the model saved in the database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As for Keras, you can access the id of the model to load it. 
Note that the app will change backend to the one used to define the model you're loading.

.. code-block:: python

    print(expe.mod_id)
    print(expe.data_id)

    expe.load_model(expe.mod_id, expe.data_id)


You can now predict with your model.

.. code-block:: python

    expe.predict(data['X'])





.. _Keras: http://keras.io/
.. _`scikit-learn`: http://scikit-learn.org/stable/


