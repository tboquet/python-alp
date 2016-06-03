============
Requirements
============

Because the whole architecture has a lot of components we use Docker_ to manage the platform.

You should first `install Docker`_ and `install nvidia-docker`_.

Using Docker_ and `nvidia-docker`_ is recommanded because we don't cover the setup of all the services on one host. The use of CUDA inside docker container makes the configuration a lot easier.

.. _Docker: https://www.docker.com/
.. _`nvidia-docker`: https://github.com/NVIDIA/nvidia-docker
.. _`install Docker`: https://docs.docker.com/engine/installation/linux/ubuntulinux/
.. _`install nvidia-docker`: https://github.com/NVIDIA/nvidia-docker#quick-start
