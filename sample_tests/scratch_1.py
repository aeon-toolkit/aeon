from aeon.similarity_search import SeriesSearch
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

dtype='int64'
apply_exclusion_to_result=False

X = np.asarray(
    [[[1, 2, 3, 4, 5, 6, 7, 8]], [[1, 2, 4, 4, 5, 6, 5, 4]]], dtype=dtype
)
S = np.asarray([[3, 4, 5]], dtype=dtype)
L = 3

search = SeriesSearch(k=2, inverse_distance=True)
search.fit(X)
mp, ip = search.predict(S, L)

print(mp,"\n \n",ip)
print(type(mp),type(ip))

print(assert_almost_equal(mp[0][:],[0.19245009, 0.28867513]))
print(assert_array_equal(ip[0][:],[[0, 5],[0, 0]]))