Monte Carlo API
~~~~~~~~~~~~~~~

Monte Carlo API applies to a series of methods that apply to a 3 dimensional data cube where the dimensions represent the time, trials and assets respectively. For example, if the simulated cube projects 10 years of monthly data for 8 assets for 10000 trials (that is 10000 simulations), the cube will be a numpy array with shape (120, 10000, 8).

.. toctree::

    Returns <returns/index>
    Risk <risk/index>
    Sensitivity <sensitivity/index>
