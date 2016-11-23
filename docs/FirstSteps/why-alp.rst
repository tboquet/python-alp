========
Why ALP?
========

We noticed that, when dealing with a Machine Learning problem, we sometime spend more time dealing with model/architecture selection than actually fitting the models. To help that process, we developed an Asynchronous Learning Platform (ALP) that uses the hardware (CPU+GPU) at a convenient capacity. That platform relies on separated services running on Docker containers. Some container deals with the asynchronous fits, some interface with the databases where the parameters of the models are stored, another one sends messages to the other containers... The whole thing runs in the background so that the final user (typically a Data Scientist that want to launch several experiments at the same time) does not directly interact with the databases or the broker and just runs code in the usual Jupyter Notebook. 

================================
What kind of models can you use?
================================

So far, the whole Keras_ neural network library is supported, as well as several models from the `scikit-learn`_ library. 


==============================================
What do I need to run ALP? What is inside ALP?
==============================================

You need a Linux machine to run ALP [1]_.
ALP relies heavily on Docker, RabbitMQ, Celery, MongoDB and Jupyter Notebook. It also supports interfacing with Fuel thus depends on Theano. It's implemented in Python. However since all services runs into Docker containers, your OS only needs Docker (and nvidia-docker if you want to use a NVIDIA GPU). The CLI will help you do the setup and launch a config in no time.

All of this concepts and dependencies are explained later in the Setup and Userguide sections.


======================
How could ALP help me?
======================

We believe it might be useful for several applications such as:

- **hyperparameters tuning**: for instance if you want to test several architectures on your neural network model, ALP can help you in dealing with the tedious task of logging all the architectures, parameters and results. They are all automatically stored in the databases and you just have to select the best model given the validation(s) you specified.
- **post analysis of several models**: 
- **fitting several models on several data streams**: 


.. [1] unfortunately at the time of the development, running MongoDB in a Windows Docker was not a possibility, but we will check out that soon.


.. _Keras: http://keras.io/
.. _`scikit-learn`: http://scikit-learn.org/stable/