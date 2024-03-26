#!/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"

excluded=(
  "examples/datasets/load_data_from_web.ipynb"
  "examples/benchmarking/reference_results.ipynb"
  "examples/benchmarking/bakeoff_results.ipynb"
  "examples/benchmarking/regression.ipynb"
)

shopt -s lastpipe
notebooks=()
runtimes=()

# Loop over all notebooks in the examples directory.
find "examples/" -name "*.ipynb" -print0 |
  while IFS= read -r -d "" notebook; do
    # Skip notebooks in the excluded list.
    if printf "%s\0" "${excluded[@]}" | grep -Fxqz -- "$notebook"; then
      echo "Skipping: $notebook"
    # Run the notebook.
    else
      echo "Running: $notebook"

      start=$(date +%s)
      $CMD "$notebook"
      end=$(date +%s)

      notebooks+=("$notebook")
      runtimes+=($((end-start)))
    fi
  done

# print runtimes and notebooks
echo "Runtimes:"
paste <(printf "%s\n" "${runtimes[@]}") <(printf "%s\n" "${notebooks[@]}")
