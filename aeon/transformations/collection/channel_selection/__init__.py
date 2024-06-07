"""Channel selection transformations.

Channel selection transformers select a subset of channels for a collection by a
method described in fit (if supervised), then return only those channels for a
collection using transform.
"""

__all__ = ["ElbowClassPairwise", "ElbowClassSum"]


from aeon.transformations.collection.channel_selection._elbow_class import (
    ElbowClassPairwise,
    ElbowClassSum,
)
