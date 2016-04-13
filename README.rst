========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |requires|
        | |coveralls| |codecov|
        | |codacy| |codeclimate|

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

.. end-badges

Machine learning for teams

* Free software: BSD license

Installation
============

::

    pip install alp

Documentation
=============



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
