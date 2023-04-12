#!/bin/bash

# Helper script to download and install aeon from test PyPI to check wheel
# and upload prior to new release

set -e

# Version to test, passed as input argument to script
VERSION=$1

# Make temporary directory
echo "Making test directory ..."
mkdir "$HOME"/testdir
cd "$HOME"/testdir

# Create test environment
echo "Creating test environment ..."

# shellcheck disable=SC1091
source "$(conda info --base)"/etc/profile.d/conda.sh  # set up conda
conda create -n aeon_testenv python=3.9
conda activate aeon_testenv

# Install from test PyPI
echo "Installing aeon from Test PyPI ..."
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple aeon=="$VERSION"
echo "Successfully installed aeon from Test PyPI."

# Clean up test directory and environment
echo "Cleaning up ..."
conda deactivate
conda remove -n aeon_testenv --all -y
rm -r "$HOME"/testdir

echo "Done."
