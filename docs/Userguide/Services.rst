========
Services
========

In this section we describe the different services (such as the Jupyter Notebook, RabbitMQ, the Models databases ...) running in separated Docker containers (resp. the Controller, the Broker, Mongos Models ...). As we tried to separate the services as much as possible, sometimes the container is assimilated to the service. 

Controller
~~~~~~~~~~

The Controller is the user endpoint of the library. It serves a Jupyter notebook in which the user sends the commands (such as `import alp`). It is linked to all other containers. 

Mongo Models
~~~~~~~~~~~~

Mongo Models is a container that runs a MongoDB service in which the architecture of the models that are trained through ALP are saved.


Mongo Results
~~~~~~~~~~~~~

Broker
~~~~~~

Also called scheduler, hhe distributes the tasks

Worker(s)
~~~~~~~~~

he runs computations

Job monitor
~~~~~~~~~~~

he distributes the tasks
