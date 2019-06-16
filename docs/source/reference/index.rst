API Reference
=============

The exact API of all functions and classes, as given by the docstrings. The API documents expected types and allowed features for all functions, and all parameters available for the algorithms.

Core
----

Core API applies to a series of methods that apply to a pandas :code:`DataFrame` or :code:`Series` or iterable TimeSeries like object.


.. toctree::
    :maxdepth: 3

    core/index

Monte Carlo
-----------

Monte Carlo API applies to a series of methods that apply to a 3 dimensional data cube where the dimensions represent the time, trials and assets respectively. For example, if the simulated cube projects 10 years of monthly data for 8 assets for 10000 trials (that is 10000 simulations), the cube will be a numpy array with shape (120, 10000, 8).

.. toctree::
    :maxdepth: 3

    monte_carlo/index

Datasets
--------

Data sets contain a series of python data objects to help get the user started on the package.

.. toctree::
    :maxdepth: 2

    datasets/index
