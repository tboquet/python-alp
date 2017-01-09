=====================================================
Tutorial 2 :  Feed simple data to your ALP Experiment
=====================================================

In this tutorial, we will build an Experiment with a simple model and
fit it on various number of pieces of data The aim of this tutorial is
to explain the expected behaviour of ALP.

1 - Get some data
~~~~~~~~~~~~~~~~~

Let us start with the usual Iris dataset.

.. code:: python

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # get some data
    iris = datasets.load_iris()
    X_train, X_val, y_train, y_val = train_test_split(
                        iris.data, iris.target, test_size=100, random_state=0)

The data is then put in the form ALP expects: a dictionary with a field
'X' for the input and a field 'y' for the output. Note that the same is
done for the validation data.

.. code:: python

    data, data_val = dict(), dict()
    data["X"], data["y"] = X_train,  y_train
    data_val["X"], data_val["y"] = X_val, y_val

Let us shuffle the data some more. After these lines, 2 more datasets
are created.

.. code:: python

    more_data, some_more_data = dict(), dict()
    more_data["X"], some_more_data["X"], more_data["y"], some_more_data["y"] = train_test_split(
                        iris.data, iris.target, test_size=75, random_state=1)

2 - Expected behaviour with sklearn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2.1 - Defining the experiment and model
+++++++++++++++++++++++++++++++++++++++

We then define a first simple sklearn logistic regression.

.. code:: python

    from alp.appcom.core import Experiment
    from  sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression()
    Expe = Experiment(lr)

2.2 - Fitting with one data set and one validation
++++++++++++++++++++++++++++++++++++++++++++++++++

Fitting one data set with one validation set is done this way:

.. code:: python

    Expe.fit([data],[data_val])


.. parsed-literal::

    ({'data_id': '1c59c0c562a5abdb84ad4f4a2c1868bf',
      'metrics': {'iter': nan,
       'score': [0.97999999999999998],
       'val_score': [0.93999999999999995]},
      'model_id': '5cabd17bbac6934fb487fa7f69bbda6e',
      'params_dump': u'/parameters_h5/5cabd17bbac6934fb487fa7f69bbda6e1c59c0c562a5abdb84ad4f4a2c1868bf.h5'},
     None)



Now let's take a look at the results: 

* there is a data\_id field: that is where the data is stored in the appropriate collection. 

* there is a model\_id field: this is where the model architecture is stored. 

*  theparam\_dump field is path of a file where the *attributes* of the fitted model are stored. 

* the metrics field is itself a dictionary with several attributes: 
   * the iter field is here for compatibility with the keras backend. 

   * the score field is model specific, you will have to look into sklearn's documentation to see what kind of metric is used. For the logistic regression, it is the accuracy. This field is then the accuracy of the fitted model on the training data. 

   * the val\_score is the score on the validation data (it is still the accuracy in this case).


You can access the full result of the experiment in the full\_res
attribut of the object.

.. code:: python

    Expe.full_res


.. parsed-literal::

    {'data_id': '1c59c0c562a5abdb84ad4f4a2c1868bf',
     'metrics': {'iter': nan,
      'score': [0.97999999999999998],
      'val_score': [0.93999999999999995]},
     'model_id': '5cabd17bbac6934fb487fa7f69bbda6e',
     'params_dump': u'/parameters_h5/5cabd17bbac6934fb487fa7f69bbda6e1c59c0c562a5abdb84ad4f4a2c1868bf.h5'}



Predicting the "more\_data" on the model fitted on "data" is done this
way.

.. code:: python

    pred_on_more_data = Expe.predict(more_data["X"])

At this point, pred\_on\_more\_data is a vector of prediction. It's
accuracy is obtained as follows:

.. code:: python

    accuracy_score(pred_on_more_data,more_data["y"])




.. parsed-literal::

    0.95999999999999996



Now you can check that the full\_res field of the Expe object was not
modified during the predict call.

.. code:: python

    Expe.full_res




.. parsed-literal::

    {'data_id': '1c59c0c562a5abdb84ad4f4a2c1868bf',
     'metrics': {'iter': nan,
      'score': [0.97999999999999998],
      'val_score': [0.93999999999999995]},
     'model_id': '5cabd17bbac6934fb487fa7f69bbda6e',
     'params_dump': u'/parameters_h5/5cabd17bbac6934fb487fa7f69bbda6e1c59c0c562a5abdb84ad4f4a2c1868bf.h5'}



2.3 - Fitting with one data set and no validation:
++++++++++++++++++++++++++++++++++++++++++++++++++

If you want to fit an experiment and don't have a validation set, you
need to specify a None in the validation field. Note that all the fields
have changed. Since the data has changed, the data\_id is different. The
model created is a new one, so are the parameters. Finally, the metrics
are different.

.. code:: python

    Expe.fit([some_more_data],[None])




.. parsed-literal::

    ({'data_id': '3554c1421fd9056e69c3cdf1b0ec8c3f',
      'metrics': {'iter': nan, 'score': [0.95999999999999996], 'val_score': [nan]},
      'model_id': 'ceb5d5632334515c4ebbd72a256bd421',
      'params_dump': u'/parameters_h5/ceb5d5632334515c4ebbd72a256bd4213554c1421fd9056e69c3cdf1b0ec8c3f.h5'},
     None)



As a result, the model actually stored in the Experiment at that time of
the code execution is not the same as in 2.2. You can check that by
predicting on the more\_data set and check that the score is not the
same.

.. code:: python

    pred_on_more_data = Expe.predict(more_data["X"])
    accuracy_score(pred_on_more_data,more_data["y"])




.. parsed-literal::

    0.94666666666666666



2.4 - Fitting several dataset
+++++++++++++++++++++++++++++

Now it's an important point since the behavior of sklearn differs from
the keras one: if you feed different datasets to an Experiment with an
sklearn model, ALP proceeds as such: 

* the first model is fitted, then the score and validation score are computed (on the first validation data, if provided). 

* the second model is fitted, then the score and validation score are computed (on the second validation data, if provided). 

* and so on

As a result, the parameters data\_id, model\_id and param\_dumps in the
full\_res field of the Experiment of the following line are the one of
the second model. The metrics (score and val\_score) fields have a
length of 2, one for each model.

Note that you can specify a None as validation set if you don't want to
validate a certain model.

.. code:: python

    Expe.fit([data,more_data],[None,some_more_data])


.. parsed-literal::

    ({'data_id': '2767007837282c3da5a86cfe41b57cce',
      'metrics': {'iter': nan,
       'score': [0.97999999999999998, 0.94666666666666666],
       'val_score': [nan, 0.92000000000000004]},
      'model_id': 'c6f885968087dc779ce47f3f1af86a9b',
      'params_dump': u'/parameters_h5/c6f885968087dc779ce47f3f1af86a9b2767007837282c3da5a86cfe41b57cce.h5'},
     None)
