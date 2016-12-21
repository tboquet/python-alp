================================================================================
Tutorial 1 : Simple Hyperparameter Tuning with ALP - sklearn models
================================================================================

In this tutorial, we will get some data, build an Experiment with a
simple model and tune the parameters of the model to get the best
performance on validation data (by launching several experiments). We
will then reuse this best model on unseen test data an check that it’s
better than the untuned model. The whole thing will be using the
asynchronous fit to highlight the capacity of ALP.

1 - Get some data
~~~~~~~~~~~~~~~~~~~~~

Let us start with the usual Iris dataset. Note that we will split the
test set in 2 samples of size 25: the "validation" set to select the
best model, and the "new" set to assess that the selected model was the
best.

.. code:: python

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    
    # get some data
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=50, random_state=0)
    X_test_val, X_test_new, y_test_val, y_test_new = train_test_split(
        X_test, y_test, test_size=25, random_state=1)
    
    # put it in ALP expected format
    data, data_val, data_new = dict(), dict(), dict()
    data["X"], data["y"] = X_train,  y_train
    data_val["X"], data_val["y"] = X_test_val, y_test_val
    data_new["X"], data_new["y"] = X_test_new, y_test_new

2 - Define an easy model and an ALP Experiment in a loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the sake of simplicity, we will define a simple Logistic Regression,
as the aim is not ultimate performance but demonstrate the capacities of
ALP.

Let us first define a helper function.

.. code:: python

    import random
    import sklearn.linear_model
    from alp.appcom.core import Experiment
    from operator import mul
    
    def grid_search(data, data_val, grid_dict, tries, async = False, model_type = 'LogisticRegression'):
        ''' This function randomly tries hyperparameters in Experiments and returns the best model.
        
        Args:    
            data(dict) : data to be passed as a dictionary mapping
                inputs names to np.arrays
            data_val(dict) : same for validation
            grid_dict(dict) : hyperparameter grid from which to draw samples from
            tries(int) : number of model to be generated and tested
            async(bool) : should the fit be asynchronous
            model_type(string) : type of model to be tested (must be in sklearn.linear_model)
        
        Returns:
            Expe (Experiment): an Experiment with (one of) the best model
  
        '''
        
        dict_res={}
        
        # 1 - infos
        size_grid = reduce(mul,[len(v) for v in grid_dict.values()])
        print("grid size : "+ str(size_grid))
        print("tries : "+ str(tries))
        print
        
        
        # 2 - main loop
        for i in range(tries):
            select_params =  {}
            for k,v in grid_dict.items():
                select_params[k] = random.choice(v)
            model = getattr(sklearn.linear_model, model_type)(**select_params)
            Expe = Experiment(model)
            if async:
                _, thread = Expe.fit_async([data], [data_val])
                thread.join()            
            else:
                Expe.fit([data], [data_val])
            dict_res[Expe.mod_id]=Expe.full_res["metrics"]["val_score"][0]
        
        # 3 - result extraction
        results = dict_res.values()
        best_score = max(results)
        best_model_id = [k for k in dict_res.keys() if dict_res[k]==best_score][0] 
        
        # 4 - load the model in the last experiment
        Expe.load_model(best_model_id, Expe.data_id)
        
        # 5 - verbose
        print("the best parameters are:")
        for k,v in grid_dict.items():
              if k in Expe.model_dict['model_arch']:
                print("    "+str(k)+" : "+str(Expe.model_dict['model_arch'][k]))
       
        return(Expe)
        
This helper function randomly (uniform, independent) samples hyperparameters combinations then fits the models within an ALP Experiment. It finally returns an Experiment where the best_model is loaded.


Details of what this function does is:
1. display some infos about the size of the grid.
2. main loop: as many times as `tries`, it selects randomly a point in the hyperparameter grid, creates an Experiment object with the model parametrized with this point, fits the experiment (possibly asynchronously).
3. extracts the result, note that we take the first element of the list of the keys that are associated to the best score, there is no mandatory uniqueness.  
4. load the model in the last Experiment object. The `load_model` function wipes the results of the previous model in the Experiment. Note that the `load_model` function needs the id of the data. 
5. display some verbose about the best parameters.

3 - Run the random search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # setting the seed for reproducibility: feel free to change it
    random.seed(12345)
    
    # defining the grid that will be explored
    grid_tol = [i*10**-j for i in (1,2,5) for j in (1,2,3,4,5,6)]
    grid_C = [i*10**-j for i in (1,2,5) for j in (-2,-1,1,2,3,4,5,6)]
    grid = {'tol':grid_tol,'C':grid_C}
    
    tries = 100
    
    Expe_best = grid_search(data,data_val,grid,tries,async = False)



.. parsed-literal::

    grid size : 432
    tries : 100
    
    the best parameters are:
        C : 50
        tol : 0.0001


A word on the interpretation of the params: 
 * the parameter C is the regularisation parameter of the Logistic Regression. A small value of C means a higher L2 constraint on w (the L2 constraint is not applied on $c$, the intercept parameter). A larger C can lead to overfitting, while a smaller value can lead to too much regularization. As such, it is the ideal candidate for automatic tuning.
 * the tol parameter is the tolerance for stopping criteria. Our experiments did not show a strong impact of this parameter unless it was set to high values.

4 - Validation that the best model is better than the untuned one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ALP makes prediction with the loaded best model on the unseen data easy.
The accuracy of the best model is decent (one mistake over 25 points).

.. code:: python

    pred_best_new = Expe_best.predict(X_test_new)
    print(sklearn.metrics.accuracy_score(pred_best_new,data_new["y"]))


.. parsed-literal::

    0.96

We can now create an untuned model (C=1 by default) and assess its precision on unseen data is lower that the tuned one.

.. code:: python

    model = sklearn.linear_model.LogisticRegression()
    Expe = Experiment(model)
    Expe.fit([data],[data_val])
    pred_worst_new = Expe.predict(X_test_new)
    print(sklearn.metrics.accuracy_score(pred_worst_new,data_new["y"]))


.. parsed-literal::

    0.88
