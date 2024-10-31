.. _similarity_search_ref:

Similarity search
=================

The :mod:`aeon.similarity_search` module contains algorithms and tools for similarity
search tasks.


Similarity search estimators
----------------------------

.. currentmodule:: aeon.similarity_search

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    QuerySearch
    SeriesSearch
    BaseSimiliaritySearch


Distance profile functions
--------------------------

.. currentmodule:: aeon.similarity_search.distance_profiles

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    naive_distance_profile
    normalized_naive_distance_profile
    euclidean_distance_profile
    normalized_euclidean_distance_profile
    squared_distance_profile
    normalized_squared_distance_profile

Matrix profile functions
--------------------------

.. currentmodule:: aeon.similarity_search.matrix_profiles

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    naive_matrix_profile
    stomp_normalized_euclidean_matrix_profile
    stomp_euclidean_matrix_profile
    stomp_normalized_squared_matrix_profile
    stomp_squared_matrix_profile
