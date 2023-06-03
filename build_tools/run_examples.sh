#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

find "examples/" -name "*.ipynb" -print0 |
  while IFS= read -r -d '' notebook; do
    echo "Running: $notebook"
    $CMD "$notebook"
  done
