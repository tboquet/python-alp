===========================================
Tutorial 1 : Simple hyperparameter tuning
===========================================

In this tutorial, we will get some data, build an Experiment with a simple model and tune the parameters of the model to get the best performance on validation data (by launching several experiments). We will then reuse this best model on unseen test data an check that it's better than the untuned model. The whole thing will be using the asynchronous fit to highlight the capacity of ALP.

