=========================
Set up of your containers
=========================

The library takes advantage of the modularity of Docker. For now, you have to define your architecture and launch your containers before accessing your controler using docker's command line interface.

To launch all the required containers so that the base config file works launch::

    docker run --name mongo_models -v /opt/data/mongo_data/models:/data/db -d --restart=always mongo

    docker run --name mongo_results -v /opt/data/mongo_data/results:/data/db -d --restart=always mongo

    docker run -d -v /opt/data/rabbitmq/dev/log:/dev/log -v /opt/data/rabbitmq:/var/lib/rabbitmq \
        --name=rabbitmq_sched -p 8080:15672 -p 5672:5672\
        --restart=always rabbitmq:3-management

    NV_GPU=0 nvidia-docker run -d \
        -v /path/to/your/notebooks:/notebooks -v /.theano:/root/.theano \
        -v /path/to/models/parameters/parameters_h5:/parameters_h5 \
        -v /path/to/data_generator:/data_generator --link mongo_models:mongo_m \
        --link mongo_results:mongo_r --link rabbitmq_sched:rabbitmq \
        --name ipy_controler_th -p 444:8888 --restart always tboquet/full7hc5controleralp

    NV_GPU=0 nvidia-docker run -d \
        -v /path/to/models/parameters/parameters_h5:/parameters_h5 -v /.theano:/root/.theano \
        --link=mongo_models:mongo_m --link=rabbitmq_sched:rabbitmq --link=mongo_results:mongo_r \
        --name=the_worker_a --restart=always -h keras1 tboquet/the7hc5workeralpk

    docker run -d -v /path/to/models/parameters/parameters_h5:/parameters_h5 -v /.theano:/root/.theano \
        -v /path/to/data_generator:/data_generator \
        --link=mongo_models:mongo_m --link=rabbitmq_sched:rabbitmq --link=mongo_results:mongo_r \
        --name=worker_b --restart=always -h sklearn1 tboquet/full7hc5workeralpsk


To learn more about how docker containers work, you could take a look at `Dockers's documentation`_.
Note that you could also use the `nvidia-docker` command to launch the GPU containers. You could find more information on the wiki_.

Verify that you have the five core containers working using:

.. code-block:: bash

   docker ps


You should see something like this:

.. code-block:: bash

    ece5ccb62f13        f84651421baf             "/usr/bin/tini -- cel"   1 days ago          Up 1 days           8888/tcp                                                                                    the_worker_a
    5886e73ab6dd        tboquet/the7hc5workeralp          "/usr/bin/tini -- /bi"   3 hours ago         Up 3 hours          8888/tcp                                                                           the_worker_b
    b4dcced0fc42        ca27cf26699a             "/usr/bin/tini -- ipy"   1 days ago          Up 1 days           0.0.0.0:444->8888/tcp                                                                       ipy_controler_th
    e3d6e4767d1d        rabbitmq:3-management    "/docker-entrypoint.s"   1 days ago          Up 1 days           4369/tcp, 5671/tcp, 15671/tcp, 25672/tcp, 0.0.0.0:5672->5672/tcp, 0.0.0.0:8080->15672/tcp   rabbitmq_sched
    666eae40395f        mongo                    "/entrypoint.sh mongo"   1 days ago          Up 1 days           27017/tcp                                                                                   mongo_results
    264bb9b23ea9        mongo                    "/entrypoint.sh mongo"   1 days ago          Up 1 days           27017/tcp                                                                                   mongo_models

The first container is a worker that will execute keras related tasks, the second container is a worker that will execute scikit learn related tasks, the third container is a controler, the fourth container is the broker and the two last containers are the databases.

You can also launch monitoring containers:

.. code-block:: bash

    docker run -d -p 5555:5555 -v ~/scheduler/proj:/sched \
          --link rabbitmq_sched:rabbitmq --name=flower_monitor --restart=always tboquet/anaceflo

    docker run -d --link mongo_results:mongo -p 8081:8081 --name=mongo_r_monitor --restart=always mongo-express
    docker run -d --link mongo_models:mongo -p 8082:8081 --name=mongo_m_monitor --restart=always mongo-express

The first container runs flower_, a simple application to monitor the broker, the last two containers serve web applications for each of the mongodb containers we launched earlier.

The architecture could be resumed as:

.. image:: _static/architecture.svg
            :width: 700


Add a controler
###############

If you want to add a Jupter Notebook to send models to the system, you have to choose an available port number and a new name to use::

    NV_GPU=0 nvidia-docker run -d \
        -v /path/to/your/notebooks:/notebooks -v /.theano:/root/.theano \
        -v /path/to/models/parameters/parameters_h5:/parameters_h5 \
        -v /path/to/data_generator:/data_generator --link mongo_models:mongo_m \
        --link mongo_results:mongo_r --link rabbitmq_sched:rabbitmq \
        --name new_name -p new_port:8888 --restart always tboquet/full7hc5controleralp

The controler should be available on the port `new_port` of the host.


Add a worker
############

To launch an additionnal worker that will consume in the `keras` queue you can use::

    NV_GPU=0 nvidia-docker run -d \
        -v /path/to/your/notebooks:/notebooks -v /.theano:/root/.theano \
        -v /path/to/models/parameters/parameters_h5:/parameters_h5 \
        -v /path/to/data_generator:/data_generator --link mongo_models:mongo_m \
        --link mongo_results:mongo_r --link rabbitmq_sched:rabbitmq \
        --name new_name -p new_port:8888 --restart always tboquet/full7hc5workeralpk


To launch an additionnal worker that will consume in the `sklearn` queue you can use::

    NV_GPU=0 nvidia-docker run -d \
        -v /path/to/your/notebooks:/notebooks -v /.theano:/root/.theano \
        -v /path/to/models/parameters/parameters_h5:/parameters_h5 \
        -v /path/to/data_generator:/data_generator --link mongo_models:mongo_m \
        --link mongo_results:mongo_r --link rabbitmq_sched:rabbitmq \
        --name new_name -p new_port:8888 --restart always tboquet/full7hc5workeralpsk


.. _flower: http://flower.readthedocs.io/en/latest/
.. _`Dockers's documentation`: https://docs.docker.com/engine/reference/run/
.. _wiki: https://github.com/NVIDIA/nvidia-docker/wiki

