===================================================================
Tutorial 0 : how to launch a basic experiment with keras or sklearn
===================================================================

Step 1 : launching alp
~~~~~~~~~~~~~~~~~~~~~~


Follow the instructions in the setup section.
We assume at this point that you have a Jupyter notebook running on the controller.


Step 2 : defining your model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can follow one of 2.1 or 2.2 steps regarding if you want a Keras_ or a `scikit-learn`_ backend. In both case we will do the right imports, get or generate some classification data, put them in the ALP format and generate a model. The important thing at the end of step 2 is to have the :code:`data`, :code:`data_val` and :code:`model` objects ready.

Step 2.1 : Keras
+++++++++++++++++

The following code gets some data and declares a simple artificial neural network with Keras:

.. code-block:: python

    # some imports
    from keras.layers import Dense
    from keras.layers import Input
    from keras.models import Sequential
    from keras.utils import np_utils
    from keras.utils.test_utils import get_test_data

    # some hyperparameters of the model
    input_dim = 2
    nb_hidden = 4
    nb_class = 2
    batch_size = 5
    train_samples = 20
    test_samples = 20

    # get some data
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                                nb_test=test_samples,
                                                input_shape=(input_dim,),
                                                classification=True,
                                                nb_class=nb_class)

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)

    # put the data in the form ALP expects
    data, data_val = dict(), dict()
    data["X"] = X_tr
    data["y"] = y_tr
    data_val["X"] = X_te
    data_val["y"] = y_te

    # finally define and compile the model
    model = Sequential()
    model.add(Dense(nb_hidden, input_dim=input_dim, activation='relu'))
    model.add(Dense(nb_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


Note that we compile the model so that we also have information about the optimizer.


Step 2.2 : `scikit-learn`_
+++++++++++++++++++++++++++

The following code gets some data and declares a simple logistic regression with `scikit-learn`:

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

You have access to two methods to fit the model.

* The :code:`fit` method allows you to fit the model in the same process.

	For the `scikit-learn` backend, you can launch the computation with the following command without extra arguments:

	.. code-block:: python

	    expe.fit([data], [data_val])

	Note that the :code:`data` and the :code:`data_val` are put in lists.


	With Keras you might want to specify the number of epochs and the batch_size, as you would have done to fit directly a Keras :code:`model` object. These arguments will flow trough to the final call. Note that they are not necessary for the fit, see the default arguments in the `Keras model doc <https://keras.io/models/model/>`_.

	.. code-block:: python

	    expe.fit([data], [data_val], nb_epoch=2, batch_size=batch_size)

	In both cases, the model is trained and automatically saved in the databases.

* The :code:`fit_async` method sends the model to the broker container that will manage the training using the workers you defined in the setup phase. The commands are then straightforward:
	For the `scikit-learn` backend:

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