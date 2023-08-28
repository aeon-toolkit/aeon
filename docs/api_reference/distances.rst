.. _distances_ref:

Time series distances
=====================

The :mod:`aeon.distances` module contains time series specific distance functions
that can be used in aeon and scikit learn estimators. It also contains tools for
extracting the alignment paths for a distance calculation between two series, and
tools for finding all pairwise distances


Dynamic Time Warping (DTW)
--------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    dtw_distance
    dtw_pairwise_distance
    dtw_cost_matrix
    dtw_alignment_path

Move-Split-Merge (MSM)
----------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    msm_distance
    msm_pairwise_distance
    msm_cost_matrix
    msm_alignment_path

Time Warp Edit (TWE)
--------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    twe_distance
    twe_pairwise_distance
    twe_cost_matrix
    twe_alignment_path

Weighted Dynamic Time Warping (DTW)
-----------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    wdtw_distance
    wdtw_pairwise_distance
    wdtw_cost_matrix
    wdtw_alignment_path

Derivative Dynamic Time Warping (DDTW)
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    ddtw_distance
    ddtw_pairwise_distance
    ddtw_cost_matrix
    ddtw_alignment_path

Weighted Derivative Dynamic Time Warping (WDDTW)
------------------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    wddtw_distance
    wddtw_pairwise_distance
    wddtw_cost_matrix
    wddtw_alignment_path

Longest Common Subsequence (LCSS)
---------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    lcss_distance
    lcss_pairwise_distance
    lcss_cost_matrix
    lcss_alignment_path

Edit Real Penalty (ERP)
-----------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    erp_distance
    erp_pairwise_distance
    erp_cost_matrix
    erp_alignment_path

Edit distance for real sequences (EDR)
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    edr_distance
    edr_pairwise_distance
    edr_cost_matrix
    edr_alignment_path

General methods with distance argument
--------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    distance
    pairwise_distance
    cost_matrix
    alignment_path

General methods to recover distance functions
---------------------------------------------

.. currentmodule:: aeon.distances

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    get_distance_function
    get_pairwise_distance_function
    get_cost_matrix_function
    get_alignment_path_function
