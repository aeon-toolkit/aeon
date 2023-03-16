.. _utils_ref:

Utility functions
=================

``sktime`` has a number of modules dedicated to utilities:

* :mod:`aeon.datatypes`, which contains utilities for data format checks and conversion.
* :mod:`aeon.pipeline`, which contains generics for pipeline construction.
* :mod:`aeon.registry`, which contains utilities for estimator and tag search.
* :mod:`aeon.utils`, which contains generic utility functions.


Data Format Checking and Conversion
-----------------------------------

:mod:`aeon.datatypes`

.. automodule:: aeon.datatypes
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.datatypes

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    convert_to
    convert
    check_raise
    check_is_mtype
    check_is_scitype
    mtype
    scitype
    mtype_to_scitype
    scitype_to_mtype


Pipeline construction generics
------------------------------

:mod:`aeon.pipeline`

.. automodule:: aeon.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.pipeline

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline
    sklearn_to_sktime


Estimator Search and Retrieval, Estimator Tags
----------------------------------------------

:mod:`aeon.registry`

.. automodule:: aeon.registry
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.registry

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    all_estimators
    all_tags
    check_tag_is_valid

Plotting
--------

:mod:`aeon.utils.plotting`

.. automodule:: aeon.utils.plotting
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.utils.plotting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    plot_series
    plot_lags
    plot_correlations

Estimator Validity Checking
---------------------------

:mod:`aeon.utils.estimator_checks`

.. automodule:: aeon.utils.estimator_checks
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.utils.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_estimator
