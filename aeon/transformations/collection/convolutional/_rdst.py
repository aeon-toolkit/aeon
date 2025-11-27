"""
RDST Transformer

This transformer extracts shapelets — short discriminative subsequences
from time series data.

Each extracted shapelet is stored as a dictionary with fields:

- ``values``: 2D array of shapelet values (padded). Use only the first ``length`` columns to access true values.
- ``startpoint``: Starting index of the shapelet in the original series.
- ``length``: True (unpadded) shapelet length.
- ``dilation``: Sampling interval applied during extraction.
- ``threshold``: Distance threshold used for the occurrence features.
- ``normalization``: Whether the shapelet was normalized.
- ``mean``: Mean value used for normalization.
- ``std``: Standard deviation used for normalization.
- ``class``: Class label from which the shapelet was extracted.

Detailed examples and usage can be found in the RDST shapelet
transformation example notebook.
"""
