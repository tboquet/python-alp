========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |coveralls| |codecov|
        | |codacy| |codeclimate|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/python-alp/badge/?style=flat
    :target: https://readthedocs.org/projects/python-alp
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/tboquet/python-alp.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/tboquet/python-alp

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/tboquet/python-alp?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/tboquet/python-alp

.. |requires| image:: https://requires.io/github/tboquet/python-alp/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/tboquet/python-alp/requirements/?branch=master

.. |coveralls| image:: https://coveralls.io/repos/tboquet/python-alp/badge.svg?branch=master&service=github
    :alt: Coverage Status
    :target: https://coveralls.io/r/tboquet/python-alp

.. |codecov| image:: https://codecov.io/github/tboquet/python-alp/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/tboquet/python-alp

.. |codacy| image:: https://img.shields.io/codacy/REPLACE_WITH_PROJECT_ID.svg?style=flat
    :target: https://www.codacy.com/app/tboquet/python-alp
    :alt: Codacy Code Quality Status

.. |codeclimate| image:: https://codeclimate.com/github/tboquet/python-alp/badges/gpa.svg
   :target: https://codeclimate.com/github/tboquet/python-alp
   :alt: CodeClimate Quality Status

.. |version| image:: https://img.shields.io/pypi/v/alp.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/alp

.. |downloads| image:: https://img.shields.io/pypi/dm/alp.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/alp

.. |wheel| image:: https://img.shields.io/pypi/wheel/alp.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/alp

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/alp.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/alp

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/alp.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/alp


.. end-badges

Machine learning for teams

* Free software: BSD license

Installation
============

::

    pip install alp

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
