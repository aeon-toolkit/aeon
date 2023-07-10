# -*- coding: utf-8 -*-
"""Tests of the results loaders not part of the test suite."""

from aeon.benchmarking.results_loaders import (
    NAME_ALIASES,
    get_estimator_results_as_array,
)


def check_load_all_classifiers():
    """Run through all classifiers in NAME_ALIASES."""
    for type in ["accuracy", "auroc", "balancedaccuracy", "nll"]:
        for name_key in NAME_ALIASES.keys():
            #            print(f"trying to load {name_key} for {type}")
            res, names = get_estimator_results_as_array(
                estimators=[name_key],
                include_missing=False,
                type=type,
                default_only=False,
            )
            #            print(f"{name_key} loaded {res.shape} results for {type}")
            names.sort()


#            print(f"first problem = {names[0]} last problem = {names[-1]}")


check_load_all_classifiers()
