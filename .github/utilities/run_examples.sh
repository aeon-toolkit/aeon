#!/opt/homebrew/bin/bash

# Script to run all example notebooks.
set -euxo pipefail

CMD="jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=600"
MULTITHREADED=${2:-false}

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
runtimes=()

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

if [ "$MULTITHREADED" = true ]; then
  # Detect CPU cores
  if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
  else
    CORES=$(nproc)
  fi
  echo "Running ${#notebooks[@]} notebooks in parallel on $CORES cores..."
  export CMD
  
  # Run in parallel
  printf "%s\0" "${notebooks[@]}" | xargs -0 -n 1 -P "$CORES" bash -c '$CMD "$1"' _

else
  # Sequential execution
  for notebook in "${notebooks[@]}"; do
    echo "Running: $notebook"

    start=$(date +%s)
    $CMD "$notebook"
    end=$(date +%s)

    runtimes+=($((end-start)))
  done

  # print runtimes and notebooks
  echo "Runtimes:"
  paste <(printf "%s\n" "${runtimes[@]}") <(printf "%s\n" "${notebooks[@]}")
fi
