# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=aeon
DOC_DIR=./docs
BUILD_TOOLS=./build_tools
TEST_DIR=testdir

.PHONY: help release install test lint clean dist doc docs

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

test: ## Run unit tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest

tests: test

test_softdeps: ## Run unit tests to check soft dependency handling in estimators
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}
	python -m pytest -v -n auto --showlocals --durations=20 -k 'test_all_estimators' $(PYTESTOPTIONS) --pyargs aeon.registry
	python -m pytest -v -n auto --showlocals --durations=20 -k 'test_check_estimator_does_not_raise' $(PYTESTOPTIONS) --pyargs aeon.utils
	python -m pytest -v -n auto --showlocals --durations=20 $(PYTESTOPTIONS) --pyargs aeon.tests.test_softdeps

test_softdeps_full: ## Run all non-suite unit tests without soft dependencies
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}
	python -m pytest -v --showlocals --durations=20 -k 'not TestAll' $(PYTESTOPTIONS) --ignore=aeon/utils/tests/test_mlflow_sktime_model_export.py
