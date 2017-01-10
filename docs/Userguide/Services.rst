========
Services
========

In this section we describe the different services (such as the Jupyter Notebook, RabbitMQ, the Models databases ...) running in separated Docker containers (resp. the Controller, the Broker, Mongos Models ...). As we tried to separate the services as much as possible, sometimes the container is assimilated to the service. 

Controller
~~~~~~~~~~

The Controller is the user endpoint of the library. By default, it serves a Jupyter notebook in which the user sends commands (such as `import alp`). You can also use it to run an application using ALP for either training or prediction.

Mongo Models
~~~~~~~~~~~~

Mongo Models is a container that runs a MongoDB service in which the architecture of the models that are trained through ALP are saved.


Mongo Results
~~~~~~~~~~~~~

Mongo Results is a container that runs a MongoDB service in wich the meta informations about a tasks is saved.

Broker
~~~~~~

Also called scheduler in the architecture, it distributes the tasks and gather the results.

Worker(s)
~~~~~~~~~

The workers run the tasks and send results to the MongoDB services. Each backend need at least one worker consuming from the right queue.

Job monitor
~~~~~~~~~~~

You can plug several containers to monitor jobs.


