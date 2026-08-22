# This Dockerfile is used to build aeon when launching binder.
# Find out more at: https://mybinder.readthedocs.io/en/latest/index.html

# Load jupyter python 3.12 image. Matches the Python version the notebooks are
# tested with in CI. The jupyter/* images on Docker Hub are no longer updated,
# the images are published to quay.io instead.
FROM quay.io/jupyter/scipy-notebook:python-3.12

# Set up user to avoid running as root
ARG NB_USER=user
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV HOME=/home/${NB_USER}

# Binder will automatically clone the repo, but we need to make sure the
# contents of our repo are in the ${HOME} directory
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

# graphviz is not in the base image. plot_network needs the graphviz binaries as
# well as the pydot package, as keras shells out to them to render the plot.
RUN apt-get update && \
    apt-get install --yes --no-install-recommends graphviz && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch user and directory
USER ${USER}
WORKDIR ${HOME}

# Install aeon and the dependencies required to run the example notebooks.
# llvmlite is deliberately not installed separately, pip resolves it against the
# version numba pins when aeon is installed.
RUN pip install --upgrade pip --no-cache-dir
RUN pip install .[binder] --no-cache-dir
