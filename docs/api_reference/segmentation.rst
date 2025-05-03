.. _segmentation_ref:

Segmentation
============

Time series segmentation involves partitioning a series into regions
that are dissimilar to neighboring regions. The :mod:`aeon.segmentation` module
contains algorithms and tools for time series segmentation.

.. currentmodule:: aeon.segmentation

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    AutoPlaitSegmenter
    BinSegmenter
    ClaSPSegmenter
    FLUSSSegmenter
    InformationGainSegmenter
    GreedyGaussianSegmenter
    EAggloSegmenter
    HMMSegmenter
    HidalgoSegmenter
    RandomSegmenter

Base
----

.. currentmodule:: aeon.segmentation.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseSegmenter
