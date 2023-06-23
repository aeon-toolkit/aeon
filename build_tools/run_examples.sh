#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

excluded=("examples/datasets/benchmarking_data.ipynb")

# Loop over all notebooks in the examples directory.
find "examples/" -name "*.ipynb" -print0 |
  while IFS= read -r -d "" notebook; do
    # Skip notebooks in the excluded list.
    if printf "%s\0" "${excluded[@]}" | grep -Fxqz -- "$notebook"; then
      echo "Skipping: $notebook"
    # Run the notebook.
    else
      echo "Running: $notebook"
      $CMD "$notebook"
    fi
  done
