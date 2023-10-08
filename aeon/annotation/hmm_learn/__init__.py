"""Implements wrappers for estimators from hmmlearn."""

__all__ = ["BaseHMMLearn", "GaussianHMM", "GMMHMM", "PoissonHMM"]

from aeon.annotation.hmm_learn.base import BaseHMMLearn
from aeon.annotation.hmm_learn.gaussian import GaussianHMM
from aeon.annotation.hmm_learn.gmm import GMMHMM
from aeon.annotation.hmm_learn.poisson import PoissonHMM
