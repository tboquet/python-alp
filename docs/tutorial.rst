Getting started with ALP!
=========================

Please go through the environment setup before going further. You should have your docker containers ready to accept jobs.
Verify that you have the five core containers working using:

.. code-block:: bash

   docker ps


You should see something like this:

.. code-block:: bash

    ece5ccb62f13        f84651421baf             "/usr/bin/tini -- cel"   8 days ago          Up 8 days           8888/tcp                                                                                    the_worker_a
    b4dcced0fc42        ca27cf26699a             "/usr/bin/tini -- ipy"   8 days ago          Up 8 days           0.0.0.0:444->8888/tcp                                                                       ipy_controler_th
    e3d6e4767d1d        rabbitmq:3-management    "/docker-entrypoint.s"   8 days ago          Up 8 days           4369/tcp, 5671/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:5672->5672/tcp, 0.0.0.0:8080->15672/tcp   rabbitmq_sched
    666eae40395f        mongo                    "/entrypoint.sh mongo"   8 days ago          Up 8 days           27017/tcp                                                                                   mongo_results
    264bb9b23ea9        mongo                    "/entrypoint.sh mongo"   8 days ago          Up 8 days           27017/tcp                                                                                   mongo_models

The first container is a worker, the second container is a controler, the third container is the broker and the two last containers are the databases.

For now ALP fully supports Keras and partially supports scikit-learn (linear models).

Keras example
-------------

We will begin by declaring a simple artificial neural network with Keras:

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

We then instanciate an `Experiment`:

.. code-block:: python

    from alp.appcom.core import Experiment

    expe = Experiment("keras", model)
    

Then, you have access to two methods to fit the model.
The `fit` method allows you to fit the model in the same process.

.. code-block:: python

    expe.fit([data], [data_val], custom_objects=custom_objects, nb_epoch=2,
             batch_size=batch_size)


Here, you will see the regular print output of Keras. The model is being trained and automatically saved in the database. 


The `fit_async` method send the model to the broker container that will manage the training using the workers you defined in the setup phase.

.. code-block:: python

    expe.fit_async([data], [data_val], custom_objects=custom_objects,
                   nb_epoch=2, batch_size=batch_size)


For now, we don't directly redirect the training information from the worker to a web application or a log so you can have it in real time. This feature is on the todo list and will be implemented in the following weeks.
Like for the fit method, the architecture of the model is saved in the db along with the performance and the parameters are dumped in an HDF5 file.


Predictions using the model saved in the database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the experiment has been fitted, you can access the id of the model in the db and load it to make prediction or access the parameters in the current process.

.. code-block:: python

    print(expe.model_id)
    print(expe.data_id)

    expe.load_model(expe.mod_id, expe.data_id)


It's then possible to make predictions using the loaded model.

.. code-block:: python

    expe.predict(data['X'].astype('float32'))





