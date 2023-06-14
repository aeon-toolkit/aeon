#!/bin/bash

# Script to search for missing init FILES.
set -euxo pipefail

FILES=$( find ./aeon -type d '!' -exec test -e "{}/__init__.py" ";" -not -path "**/__pycache__" -not -path "**/benchmarking/example_results*" -not -path "**/datasets/data*" -print )

if [[ -n "$FILES" ]]
then
    echo "Missing __init__.py files detected in the following modules:"
    echo "$FILES"
    exit 1
fi
