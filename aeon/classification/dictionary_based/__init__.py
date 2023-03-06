# -*- coding: utf-8 -*-
"""Dictionary based time series classifiers."""
__all__ = [
    "IndividualBOSS",
    "BOSSEnsemble",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "IndividualTDE",
    "WEASEL",
    "MUSE",
]

from aeon.classification.dictionary_based._boss import BOSSEnsemble, IndividualBOSS
from aeon.classification.dictionary_based._cboss import ContractableBOSS
from aeon.classification.dictionary_based._muse import MUSE
from aeon.classification.dictionary_based._tde import (
    IndividualTDE,
    TemporalDictionaryEnsemble,
)
from aeon.classification.dictionary_based._weasel import WEASEL
