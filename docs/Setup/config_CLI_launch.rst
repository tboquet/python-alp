==========================
Launching ALP with the CLI
==========================

To begin, we can generate a base configuration using ALP CLI. We choose to write configuration files on the host machine in order to be able to customize them easily afterwards.


.. code-block:: bash

     alp --help

Will provide you with some help about the command line interface.

Generating a new configuration is as easy as:

.. code-block:: bash

    alp --verbose genconfig --outdir=/path/to/a/directory


The command will generate a base configuration with one controler, one scikit learn worker and one keras worker.
We specify the output directory where we want to write the three configuration files. The first file :code:`alpdb.json` defines the connection between the database of models and other containers. The second file :code:`alpapp.json` defines the connections between the broker, its database and the other containers. The third file :code:`containers.json` defines all the containers of the architecture. The linking is automatically done and ALP will use the newly created files to launch a new instance.

In any case, verify that the ports that you want to use are free for the broker to communicate with the monitoring containers and for the jupyter notebooks (if any) to run.

To start all the services you can use :code:`alp service start`:

.. code-block:: bash

    alp --verbose service start /path/to/a/directory

You can then take a look at the status of the containers:

.. code-block:: bash

    alp --verbose status /path/to/a/directory


You should be able to access the Jupyter notebook on the port :code:`440` of the machine where you launched the services.
