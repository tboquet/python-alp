#!/usr/bin/env bash

# launch the mongo model docker container

echo "Building ..."
echo "Launch the MongoDB models container ..."
docker run --name mongo_models -v /opt/data/mongo_data/models:/data/db -d --restart=always mongo
echo -e "\n"

echo "Launch the MongoDB results container ..."
# launch the mongo results docker container
docker run --name mongo_results -v /opt/data/mongo_data/results:/data/db -d --restart=always mongo
echo -e "\n"

echo "Launch the Rabbitmq broker container ..."
# start the rabbitmq broker
docker run -d -v /etc/localtime:/etc/localtime:ro \
       -v /opt/data/rabbitmq/dev/log:/dev/log -v /opt/data/rabbitmq:/var/lib/rabbitmq \
       --name=rabbitmq_sched -p 8080:15672 -p 5672:5672\
       --restart=always rabbitmq:3-management
echo -e "\n"

