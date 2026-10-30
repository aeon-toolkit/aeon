.. _similarity_search_ref:

Similarity search
=================

The :mod:`aeon.similarity_search` module contains algorithms and tools for similarity
search tasks. The module is organized into two main categories:

- **Subsequence search**: Finding nearest neighbors among subsequences of time series
- **Whole series search**: Finding nearest neighbors among complete time series


Subsequence Search Estimators
------------------------------

.. currentmodule:: aeon.similarity_search.subsequence

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveSubsequenceSearch
    MASS


Whole Series Search Estimators
-------------------------------

.. currentmodule:: aeon.similarity_search.whole_series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    NaiveSeriesSearch
    SimHashIndexANN


Base Estimators
---------------

.. currentmodule:: aeon.similarity_search

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSimilaritySearch


.. currentmodule:: aeon.similarity_search.subsequence

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSubsequenceSearch
    BaseDistanceProfileSearch


.. currentmodule:: aeon.similarity_search.whole_series

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseWholeSeriesSearch
