======================
Use Fuel or generators
======================

You can easily use Fuel_ iterators in an Experiment.
We will first create some fake data.

.. code-block:: python

    import fuel
    import numpy as np
    input_dim = 2
    nb_hidden = 4
    nb_class = 2
    batch_size = 5
    train_samples = 512
    test_samples = 128
    (X_tr, y_tr), (X_te, y_te) = get_test_data(nb_train=train_samples,
                                              nb_test=test_samples,
                                              input_shape=(input_dim,),
                                              classification=True,
                                              nb_class=nb_class)

    y_tr = np_utils.to_categorical(y_tr)
    y_te = np_utils.to_categorical(y_te)

    data, data_val = dict(), dict()

    X = np.concat([X_tr, X_te])
    y = np.concat([y_tr, y_te])

    inputs = [X, X]
    outputs = [y]


We then import an helper function that will convert our list of inputs to an HDF5 dataset.
This dataset has a simple structure and we can divide it into multiple sets.


.. code-block:: python

    # we save the mean and the scale (inverse of the standard deviation)
    # for each channel
    scale = 1.0 / inputs[0].std(axis=0)
    shift = - scale * inputs[0].mean(axis=0)

    # for 3 sets, we need 3 slices
    slices = [0, 256, 512]

    # and 3 names
    names = ['train', 'test', 'valid']

    file_name = 'test_data_'
    file_path_f = to_fuel_h5(inputs, outputs, slices, names, file_name, '/data_generator')


The next step is to construct our Fuel generator using our dataset, a scheme and to transform the data so it's prepared for our model.


.. code-block:: python

    train_set = H5PYDataset(file_path_f,
                            which_sets=('train','test', 'valid'))

    scheme = SequentialScheme(examples=128, batch_size=32)

    data_stream_train = DataStream(dataset=train_set, iteration_scheme=scheme)

    stand_stream_train = ScaleAndShift(data_stream=data_stream_train,
                                       scale=scale, shift=shift,
                                       which_sources=('input_X',))


We finally build our model and wrap it in an experiment.


.. code-block:: python

    inputs = Input(shape=(input_dim,), name='X')

    x = Dense(nb_hidden, activation='relu')(inputs)
    x = Dense(nb_hidden, activation='relu')(x)
    predictions = Dense(nb_class, activation='softmax')(x)

    model = Model(input=inputs, output=predictions)

    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    expe = Experiment(model)


We can finally use the `alp.appcom.Experiment.fit_gen` method with our model and dataset.


.. code-block:: python

    expe.fit_gen([gen], [val], nb_epoch=2,
                  model=model,
                  metrics=metrics,
                  custom_objects=cust_objects,
                  samples_per_epoch=128,
                  nb_val_samples=128)

You can also use `alp.appcom.Experiment.fit_gen_async` with the same function parameters if you have a worker running.

.. code-block:: python

    expe.fit_gen([gen], [val], nb_epoch=2,
                  model=model,
                  metrics=metrics,
                  custom_objects=cust_objects,
                  samples_per_epoch=128,
                  nb_val_samples=128)


 .. _Fuel: https://fuel.readthedocs.io/en/latest/
