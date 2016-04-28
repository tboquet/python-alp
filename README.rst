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


ALP (Asynchronous Learning Platform) helps you experiment with a lot of machine learning models quickly. It provide you with a simple way of scheduling and recording experiments.

You can easily define how many workers and how many clients are available in a configuration file.


This library has been developped to work well with Keras but can suit a lot of other frameworks. 

Installation
============

::

    git clone https://github.com/tboquet/python-alp.git
    cd python-alp
    python setup.py install


Documentation
=============

https://python-alp.readthedocs.org/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox

* Free software: BSD license
