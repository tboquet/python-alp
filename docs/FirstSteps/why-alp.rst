========
Why ALP?
========

We noticed that, when dealing with a Machine Learning problem, we sometime spend more time working on building a model, testing different architectures, comparing results than actually work on the ideas that will solve our problem. To help that process, we developed an Asynchronous Learning Platform (ALP) that uses the hardware (CPU+GPU) at a convenient capacity. That platform relies on independant services running on Docker containers. For this plateform to be easy to use, we built a convenient command line interface from wich you can easily launch, stop, remove, update and monitor a configuration.

The whole system runs in the background so that the final user does not directly interact with the databases or the broker and just runs code in an usual Jupyter Notebook or from an application. You can also launch monitoring containers and access different dashboards to supervise all of your experiments. Moreover, it is possible to easily retrieve one of the trained model along with it's parameters at test time.

================================
What kind of models can you use?
================================

So far, the whole Keras_ neural network library is supported, as well as several models from the `scikit-learn`_ library. 


==============================================
What do I need to run ALP? What is inside ALP?
==============================================

You need to use a machine running Linux to use ALP [1]_.
ALP relies on Docker, RabbitMQ, Celery, MongoDB and nvidia-docker. It also supports interfacing with Fuel thus depends on Theano. It's implemented in Python. However since all services runs into Docker containers, your OS only needs Docker (and nvidia-docker if you want to use a NVIDIA GPU).

All of this concepts and dependencies are explained later in the Setup and Userguide sections.


======================
How could ALP help me?
======================

We believe it might be useful for several applications such as:

- **hyperparameters tuning**: for instance if you want to test several architectures on your neural network model, ALP can help you in dealing with the tedious task of logging all the architectures, parameters and results. They are all automatically stored in the databases and you just have to select the best model given the validation(s) you specified.
- **fitting several models on several data streams**: you have data streams coming from a source and you want to fit a lot of online models, it is easy with ALP. With the support of Fuel generators, you could transform your data on the fly. The he learning is then done using the resources of the host and the parameters of the models are stored. You could even code an API that returns prediction to your data service.
- **post analysis**: extract and explore the parameters of models given their score on several data blocks. Sometimes it could be helpful to visualise the successful set of parameters.

- **model deployment in production**: when a model is trained, you can load it and deploy it instantly in production.



.. [1] unfortunately at the time of the development, running MongoDB in a Windows Docker was not a possibility, but we will check out that soon.


.. _Keras: http://keras.io/
.. _`scikit-learn`: http://scikit-learn.org/stable/
