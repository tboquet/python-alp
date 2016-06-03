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
