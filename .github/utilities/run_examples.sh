#!/usr/bin/env bash

# Script to run all example notebooks.
set -euo pipefail

CMD="python -m jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"
MULTITHREADED=${2:-false}

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "WARNING: Windows detected. Multithreading is unstable in this environment."
    echo "Defaulting to sequential execution."
    MULTITHREADED=false
fi

excluded=(
  # try removing when 3.9 is dropped
  "examples/transformations/signature_method.ipynb"
)
if [ "$1" = true ]; then
  excluded+=(
    "examples/datasets/load_data_from_web.ipynb"
    "examples/benchmarking/published_results.ipynb"
    "examples/benchmarking/reference_results.ipynb"
    "examples/benchmarking/bakeoff_results.ipynb"
    "examples/benchmarking/regression.ipynb"
    "examples/clustering/partitional_clustering.ipynb"
    "examples/classification/hybrid.ipynb"
    "examples/classification/deep_learning.ipynb"
    "examples/classification/dictionary_based.ipynb"
    "examples/classification/distance_based.ipynb"
    "examples/classification/feature_based.ipynb"
    "examples/classification/interval_based.ipynb"
    "examples/classification/shapelet_based.ipynb"
    "examples/classification/convolution_based.ipynb"
    "examples/similarity_search/code_speed.ipynb"
  )
fi

shopt -s lastpipe
notebooks=()

# Loop over all notebooks in the examples directory.
find "examples" -name "*.ipynb" -print0 |
  while IFS= read -r -d "" notebook; do
    # Skip notebooks in the excluded list.
    if printf "%s\0" "${excluded[@]}" | grep -Fxqz -- "$notebook"; then
      echo "Skipping: $notebook"
    # Add valid notebooks to the array
    else
      notebooks+=("$notebook")
    fi
  done

LOG_DIR=$(mktemp -d)
trap 'rm -rf "$LOG_DIR"' EXIT

run_notebook() {
    local notebook=$1
    local start=$(date +%s)

    if $CMD "$notebook" > /dev/null 2>&1; then
        local status="PASS"
    else
        local status="FAIL"
        echo "::error::Failed: $notebook"
        $CMD "$notebook"
    fi

    local end=$(date +%s)
    local runtime=$((end-start))

    echo "$runtime $status $notebook" >> "$LOG_DIR/results.txt"
    echo "Finished: $notebook (${runtime}s) [$status]"
}

export -f run_notebook
export CMD
export LOG_DIR

if [ "$MULTITHREADED" = true ]; then
  # Detect CPU cores
  if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
  else
    CORES=$(nproc)
  fi
  echo "Running ${#notebooks[@]} notebooks in parallel on $CORES cores..."

  printf "%s\0" "${notebooks[@]}" | xargs -0 -n 1 -P "$CORES" bash -c 'run_notebook "$@"' _

else
  # Sequential execution
  for notebook in "${notebooks[@]}"; do
    run_notebook "$notebook"
  done
fi

FAILURES=0
if [ -f "$LOG_DIR/results.txt" ]; then
    echo -e "TIME\tSTATUS\tNOTEBOOK"
    sort -rn "$LOG_DIR/results.txt" | awk '{print $1"s\t"$2"\t"$3}'

    if grep -q "FAIL" "$LOG_DIR/results.txt"; then
        FAILURES=1
        echo ""
        echo "::error::The following notebooks FAILED:"
        grep "FAIL" "$LOG_DIR/results.txt" | awk '{print $3}'
    fi
fi

if [ $FAILURES -eq 1 ]; then
    exit 1
fi
