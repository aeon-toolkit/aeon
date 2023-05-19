# -*- coding: utf-8 -*-
"""Benchmarks tests."""
from aeon.benchmarking.results_loaders import (
    get_array_from_tsc_com,
    get_results_from_tsc_com,
)

cls = ["HC2"]
data = ["Chinatown"]


def test_get_results_from_tsc_com():
    res = get_results_from_tsc_com(classifiers=cls, datasets=data)
    assert res["HC2"]["Chinatown"] == 0.9825072886297376


def test_get_array_from_tsc_com():
    res = get_array_from_tsc_com(classifiers=cls, datasets=data)
    assert res[0][0] == 0.9825072886297376
