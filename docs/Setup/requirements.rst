============
Requirements
============

Because the whole architecture has a lot of components we use Docker_ to manage the platform and isolates the services.

ALP has been developed to run on Ubuntu and has not been tested on other OS.

You should first `install Docker`_ and `install nvidia-docker`_, then play a bit with docker (check if you can access your GPU with nvidia-docker). You then you should be ready to install ALP.

You can then get ALP via pip:
.. code:: python
	pip install git+git://github.com/tboquet/python-alp

That will install ALP on your machine, and you will be able to launch it via the Command Line Interface.

.. _Docker: https://www.docker.com/
.. _`nvidia-docker`: https://github.com/NVIDIA/nvidia-docker
.. _`install Docker`: https://docs.docker.com/engine/installation/linux/ubuntulinux/
.. _`install nvidia-docker`: https://github.com/NVIDIA/nvidia-docker/wiki/Installation

