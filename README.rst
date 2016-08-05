========
Overview
========

.. start-badges

|travis| |requires| |coveralls| |codecov| |codacy| |codeclimate| |docs|

.. |travis| image:: https://travis-ci.org/tboquet/python-alp.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/tboquet/python-alp

.. |requires| image:: https://requires.io/github/tboquet/python-alp/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/tboquet/python-alp/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/tboquet/python-alp/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/tboquet/python-alp

.. |codecov| image:: https://codecov.io/github/tboquet/python-alp/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/tboquet/python-alp

.. |codacy| image:: https://img.shields.io/codacy/b7f6d79244d8480099a3593db2de9560.svg?style=flat
    :target: https://www.codacy.com/app/tboquet/python-alp
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/tboquet/python-alp/badges/gpa.svg
   :target: https://codeclimate.com/github/tboquet/python-alp
   :alt: CodeClimate Quality Status

.. |docs| image:: https://readthedocs.org/projects/python-alp/badge/?style=flat
    :target: https://readthedocs.org/projects/python-alp
    :alt: Documentation Status

.. end-badges


ALP helps you experiment with a lot of machine learning models quickly. It provides you with a simple way of scheduling and recording experiments.

This library has been developped to work well with Keras and Scikit-learn but can suit a lot of other frameworks. 

Installation to develop your own service
========================================

::

    git clone https://github.com/tboquet/python-alp.git
    cd python-alp
    python setup.py install


Launching the services
======================

Please see the `docker setup`_ part of the documentation.


Documentation
=============

https://python-alp.readthedocs.org/

Development
===========

To run the all tests run::

    tox

If you don't have the necessary requirements installed when you run `tox` but you have `Docker` running, you can `launch all the required services` and use::

    docker run -it --rm --privileged=true --volume=/path/to/the/library/python-alp:/app --volume=~/temp/data/parameters_h5:/parameters_h5 --link=mongo_models:mongo_m --link=mongo_results:mongo_r --link rabbitmq_sched:rabbitmq --name=testenvt tboquet/pythondev

This docker container will launch sequentially `tox` and `detox`, test the library against all the supported python version and build the documentation.

Note, to combine the coverage data from all the tox environments run:

::

    PYTEST_ADDOPTS=--cov-append tox

* Free software: Apache license

.. _`docker setup`: http://python-alp.readthedocs.io/en/latest/dockersetup.html

