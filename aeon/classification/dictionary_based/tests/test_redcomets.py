"""REDCOMETS test code."""

__maintainer__ = []

# from sys import platform
#
# import numpy as np
# import pytest
#
# from aeon.classification.dictionary_based import REDCOMETS
# from aeon.datasets import load_basic_motions, load_unit_test
# from aeon.utils.validation._dependencies import _check_soft_dependencies
#
#
# @pytest.mark.skipif(
#     not _check_soft_dependencies(
#         "imbalanced-learn",
#         package_import_alias={"imbalanced-learn": "imblearn"},
#         severity="none",
#     ),
#     reason="skip test if required soft dependency imbalanced-learn not available",
# )
# def test_redcomets_score_univariate():
#     """Test of REDCOMETS train estimate on unit test data."""
#     # load unit test data
#     X_train, y_train = load_unit_test(split="train")
#     X_test, y_test = load_unit_test(split="test")
#
#     def test_variant(v, expected_result):
#         # train REDCOMETS-<v>
#         redcomets = REDCOMETS(variant=v, n_trees=3, random_state=0)
#         redcomets.fit(X_train, y_train)
#
#         score = redcomets.score(X_test, y_test)
#
#         assert isinstance(score, float)
#
#         # We cannot guarantee same results on ARM macOS
#         if platform != "darwin":
#             np.testing.assert_almost_equal(score, expected_result, decimal=4)
#
#     test_variant(1, 0.7272)
#     test_variant(2, 0.6818)
#     test_variant(3, 0.7272)
#
#
# @pytest.mark.skipif(
#     not _check_soft_dependencies(
#         "imbalanced-learn",
#         package_import_alias={"imbalanced-learn": "imblearn"},
#         severity="none",
#     ),
#     reason="skip test if required soft dependency imbalanced-learn not available",
# )
# def test_redcomets_score_multivariate():
#     """Test of REDCOMETS train estimate on unit test data."""
#     # load unit test data
#     X_train, y_train = load_basic_motions(split="train")
#     X_test, y_test = load_basic_motions(split="test")
#
#     def test_variant(v, expected_result):
#         # train REDCOMETS-<v>
#         redcomets = REDCOMETS(variant=v, n_trees=3, random_state=0)
#         redcomets.fit(X_train, y_train)
#
#         score = redcomets.score(X_test, y_test)
#
#         assert isinstance(score, float)
#
#         # We cannot guarantee same results on ARM macOS
#         if platform != "darwin":
#             np.testing.assert_almost_equal(score, expected_result, decimal=4)
#
#     test_variant(1, 0.95)
#     test_variant(2, 0.975)
#     test_variant(3, 0.975)
#     test_variant(4, 0.875)
#     test_variant(5, 0.875)
#     test_variant(6, 0.875)
#     test_variant(7, 0.875)
#     test_variant(8, 0.875)
#     test_variant(9, 0.875)
#
#
# @pytest.mark.skipif(
#     not _check_soft_dependencies(
#         "imbalanced-learn",
#         package_import_alias={"imbalanced-learn": "imblearn"},
#         severity="none",
#     ),
#     reason="skip test if required soft dependency imbalanced-learn not available",
# )
# def test_redcomets_lens_generation():
#     """Test of REDCOMETS random lens generation."""
#     # load unit test data
#     X, y = load_unit_test()
#
#     # Generate 10 random lenses
#     redcomets = REDCOMETS(random_state=0)
#     lenses = redcomets._get_random_lenses(np.squeeze(X), 10)
#
#     assert len(lenses) == 10
#     assert isinstance(lenses, list)
#
#     for w, a in lenses:
#         assert isinstance(w, int)
#         assert isinstance(a, int)
