from enum import Enum


class ClusteringAlgorithmType(Enum):
    """
    An enumeration of clustering algorithm types, dependencies, and data structures.

    This enum provides a comprehensive classification for different clustering-related
    categorizations, including algorithm types, Python dependencies, and data structures
    used in clustering processes.

    Attributes
    ----------
        Algorithm Types:
        - DISTANCE: Clustering based on distance metrics
        - DEEPLEARNING: Clustering using deep learning techniques
        - FEATURE: Clustering driven by feature extraction

        Python Dependencies:
        - TSLEARN: Time series machine learning library
        - TENSORFLOW: Deep learning framework
        - TSFRESH: Time series feature extraction library

        Data Structure Types:
        - NP_LIST: Numpy list-based data structure
        - NUMPY3D: Three-dimensional Numpy array
    """

    # Algorithm types for clustering strategies
    DISTANCE = "distance"
    DEEPLEARNING = "deeplearning"
    FEATURE = "feature"

    # Python dependencies for clustering implementations
    TSLEARN = "tslearn"
    TENSORFLOW = "tensorflow"
    TSFRESH = "tsfresh"

    # Data structure types for clustering input
    NP_LIST = "np-list"
    NUMPY3D = "numpy3D"
