==========================
Launching ALP with the CLI
==========================

To begin, we can generate a base configuration using ALP CLI. We choose to write configuration files on the host machine in order to be able to customize them easily afterwards.
Generating a new configuration is as easy as:


.. code-block:: bash

    alp --verbose genconfig --outdir=/path/to/a/directory


We specify the output directory where we want to write the three configuration files. The first file `alpdb.json` defines the connection between the database of models and other containers. The second file `alpapp.json` defines the connections between the broker, its database and the other containers. The third file `containers.json` defines all the containers of the architecture. The linking is automatically done and ALP will use the newly created files to launch a new instance.

Here we don't cover the case where another architecture with another broker is also run on the same machine. In any case, verify that the port `5672` is free for the broker to communicate with the monitoring containers.

.. code-block:: bash

    # TODO

.. _file: https://www.docker.com/
