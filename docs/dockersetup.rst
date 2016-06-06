=========================
Set up of your containers
=========================

The library takes advantage of the modularity of Docker. For now, you have to define your architecture and launch your containers before accessing your controler using docker's command line interface.

To launch all the required containers so that the base config file works launch::

    docker run --name mongo_models -v /opt/data/mongo_data/models:/data/db -d --restart=always mongo

    docker run --name mongo_results -v /opt/data/mongo_data/results:/data/db -d --restart=always mongo

    docker run -d -v /etc/localtime:/etc/localtime:ro \
        -v /opt/data/rabbitmq/dev/log:/dev/log -v /opt/data/rabbitmq:/var/lib/rabbitmq \
        --name=rabbitmq_sched -p 8080:15672 -p 5672:5672\
        --restart=always rabbitmq:3-management

    docker run -d `curl -s http://localhost:3476/v1.0/docker/cli?dev=0\&vol=nvidia_driver` \
        -v /home/tboquet/notebooks:/notebooks \
        -v /opt/data/parameters_h5:/parameters_h5 -v /opt/data/r2dbh5:/r2dbh5 --link mongo_models:mongo_m \
        --link mongo_results:mongo_r -v /home/tboquet/scheduler/proj:/sched --link rabbitmq_sched:rabbitmq \
        --name ipy_controler_th -p 444:8888 -w "/sched" --restart always tboquet/the7hc5controleralp

    docker run -d `curl -s http://localhost:3476/v1.0/docker/cli?dev=0\&vol=nvidia_driver` \
        -v /opt/data/parameters_h5:/parameters_h5 \
        -v /home/tboquet/scheduler/proj:/sched -v /opt/data/r2dbh5:/r2dbh5 \
        --link=mongo_models:mongo_m --link=rabbitmq_sched:rabbitmq --link=mongo_results:mongo_r \
        --name=the_worker_a --restart=always tboquet/the7hc5workeralp

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

You can also launch monitoring containers:

.. code-block:: bash

    docker run -d -p 5555:5555 -v /etc/localtime:/etc/localtime:ro -v ~/scheduler/proj:/sched \
          --link rabbitmq_sched:rabbitmq --name=flower_monitor --restart=always tboquet/anaceflo

    docker run -d --link mongo_results:mongo -p 8081:8081 --name=mongo_r_monitor --restart=always knickers/mongo-express
    docker run -d --link mongo_models:mongo -p 8082:8081 --name=mongo_m_monitor --restart=always knickers/mongo-express

The first container runs flower_, a simple application to monitor the broker, the last two containers serve web applications for each of the mongodb containers we launched earlier.

The architecture could be resumed as:

.. image:: _static/architecture.svg
            :width: 480
