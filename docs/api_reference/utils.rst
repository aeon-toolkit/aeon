.. _utils_ref:

Utility functions
=================

``aeon`` has a number of modules dedicated to utilities:

* :mod:`aeon.pipeline`, which contains generics for pipeline construction.
* :mod:`aeon.registry`, which contains utilities for estimator and tag search.
* :mod:`aeon.utils`, which contains generic utility functions.


Pipeline construction
---------------------

:mod:`aeon.pipeline`

.. automodule:: aeon.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.pipeline

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_pipeline
    sklearn_to_aeon


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


Estimator Validity Checking
---------------------------

:mod:`aeon.testing.estimator_checks`

.. automodule:: aeon.testing.estimator_checks
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.testing.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    check_estimator

Data Validation Tools
---------------------

:mod:`aeon.utils`

.. automodule:: aeon.utils.validation
    :no-members:
    :no-inherited-members:

.. currentmodule:: aeon.testing.estimator_checks

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    convert_collection
