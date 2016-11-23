====================================================
Tutorial 3 : Feed more data with Fuel or generators
====================================================

Because we aim at supporting online learning on streamed data, we think that generators support was a good start.
We extended the support of Fuel_, a MILA library that helps to iterate over data, pre-process it then serialize the pipeline.
You can easily use Fuel iterators in an Experiment.

.. _Fuel: https://github.com/mila-udem/fuel