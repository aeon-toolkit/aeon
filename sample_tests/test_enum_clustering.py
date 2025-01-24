# #_k_sc.py passed
# import numpy as np
# from aeon.clustering import KSpectralCentroid
# X = np.random.random(size=(10,2,20))
# clst = KSpectralCentroid(n_clusters=2, max_shift=2)
# clst.fit(X)
# KSpectralCentroid(max_shift=2, n_clusters=2)
# preds = clst.predict(X)
# print(preds)

# elastic_som.py working
# import numpy as np
# from aeon.clustering import ElasticSOM
# X = np.random.random(size=(10,2,20))
# clst = ElasticSOM(n_clusters=3, random_state=1, num_iterations=10)
# clst.fit(X)
# print(clst)
# # ElasticSOM(n_clusters=3, num_iterations=10, random_state=1)


# import numpy as np
# from aeon.clustering import TimeSeriesKMeans
# X = np.random.random(size=(10,2,20))
# clst= TimeSeriesKMeans(distance="euclidean",n_clusters=2)
# clst.fit(X)
# print(clst)
# # TimeSeriesKMeans(distance='euclidean', n_clusters=2)
# preds = clst.predict(X)
# print(preds)

# packages should be of str format 
# from aeon.clustering import TimeSeriesKernelKMeans
# from aeon.datasets import load_basic_motions
# # Load data
# X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
# X_test, y_test = load_basic_motions(split="TEST")[0:10]
# # Example of KernelKMeans Clustering
# kkm = TimeSeriesKernelKMeans(n_clusters=3, kernel='rbf')  # doctest: +SKIP
# kkm.fit(X_train)  # doctest: +SKIP
# print(kkm)
# # TimeSeriesKernelKMeans(kernel='rbf', n_clusters=3)
# preds = kkm.predict(X_test)  # doctest: +SKIP
# print(preds)


# from aeon.clustering import TimeSeriesKShape
# from aeon.datasets import load_basic_motions
# # Load data
# X_train, y_train = load_basic_motions(split="TRAIN")[0:10]
# X_test, y_test = load_basic_motions(split="TEST")[0:10]
# # Example of KShapes clustering
# ks = TimeSeriesKShape(n_clusters=3, random_state=1)  # doctest: +SKIP
# ks.fit(X_train)  # doctest: +SKIP
# print(ks)
# # TimeSeriesKShape(n_clusters=3, random_state=1)
# preds = ks.predict(X_test)  # doctest: +SKIP
# print(preds)

# from aeon.transformations.collection import Resizer
# from aeon.clustering import TimeSeriesKMeans
# from aeon.datasets import load_unit_test
# from aeon.clustering.compose import ClustererPipeline
# X_train, y_train = load_unit_test(split="train")
# X_test, y_test = load_unit_test(split="test")
# pipeline = ClustererPipeline(Resizer(length=10), TimeSeriesKMeans._create_test_instance())
# pipeline.fit(X_train, y_train)
# print(pipeline)
# # ClustererPipeline(...)
# y_pred = pipeline.predict(X_test)
# print(y_pred)

# from aeon.clustering.deep_learning import AEAttentionBiGRUClusterer
# from aeon.clustering import DummyClusterer
# from aeon.datasets import load_unit_test
# X_train, y_train = load_unit_test(split="train")
# X_test, y_test = load_unit_test(split="test")
# _clst = DummyClusterer(n_clusters=2)
# abgruc=AEAttentionBiGRUClusterer(estimator=_clst, n_epochs=20, batch_size=4) # doctest: +SKIP
# print(abgruc)
# abgruc.fit(X_train)  # doctest: +SKIP
# print(abgruc)
# AEAttentionBiGRUClusterer(...)


# import numpy as np
# from sklearn.cluster import KMeans
# from aeon.clustering.feature_based import SummaryClusterer
# X = np.random.random(size=(10,2,20))
# clst = SummaryClusterer(estimator=KMeans(n_clusters=2))
# clst.fit(X)
# print(clst)
# # SummaryClusterer(...)
# preds = clst.predict(X)
# print(preds)

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from aeon.similarity_search.query_search import QuerySearch

dtype='int64'
X = np.asarray(
    [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
)
q = np.asarray([[3, 4, 5]], dtype=dtype)

# Test with k=2, non-normalized
search = QuerySearch(k=2)
search.fit(X)
_, idx_no_exclusion = search.predict(q)
_, idx_with_exclusion = search.predict(q, apply_exclusion_to_result=True)
print(idx_with_exclusion,idx_no_exclusion,sep='--->',end='\n')

# Test with k=3, normalized and threshold
search = QuerySearch(k=3, normalise=True, threshold=2.0)
search.fit(X)
_, idx_no_exclusion = search.predict(q)
_, idx_with_exclusion = search.predict(q, apply_exclusion_to_result=True)
print(idx_with_exclusion,idx_no_exclusion,sep='--->',end='\n')

# Test with threshold only and inverse distance
search = QuerySearch(k=np.inf, threshold=1.0, inverse_distance=True)
search.fit(X)
_, idx_no_exclusion = search.predict(q)
_, idx_with_exclusion = search.predict(q, apply_exclusion_to_result=True)
# print(idx_with_exclusion,idx_no_exclusion,sep='--->',end='\n')

# Test with different exclusion factor
search = QuerySearch(k=3)
search.fit(X)
_, idx_default = search.predict(q, apply_exclusion_to_result=True)
_, idx_custom = search.predict(q, apply_exclusion_to_result=True, exclusion_factor=1.5)
# assert len(idx_default) >= len(idx_custom)

# Test with squared distance
search = QuerySearch(k=2, distance="squared", normalise=True)
search.fit(X)
_, idx_no_exclusion = search.predict(q)
_, idx_with_exclusion = search.predict(q, apply_exclusion_to_result=True)
# print(idx_with_exclusion,idx_no_exclusion,sep='--->',end='\n')