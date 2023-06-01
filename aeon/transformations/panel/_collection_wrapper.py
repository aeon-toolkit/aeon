# -*- coding: utf-8 -*-
"""Catch22 features.

A transformer for the Catch22 features.
"""

__author__ = ["MatthewMiddlehurst"]
__all__ = ["Catch22"]


from aeon.transformations.base import BaseTransformer


class CollectionTransformerWrapper(BaseTransformer):
    """
    todo
    """

    _tags = {
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
    }

    def __init__(
        self,
        features="all",
        catch24=False,
        outlier_norm=False,
        replace_nans=False,
        n_jobs=1,
    ):
