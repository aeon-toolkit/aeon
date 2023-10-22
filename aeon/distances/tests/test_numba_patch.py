import pickle

import numba.core.caching
import numpy as np
import pytest

from aeon.distances._distance import DISTANCES


# Step 1: Define a new function with the same signature as the method you're replacing.
def custom_load_index(self):
    print("HEHREH")  # noqa: T001, T201
    try:
        with open(self._index_path, "rb") as f:
            version = pickle.load(f)
            data = f.read()
    except FileNotFoundError:
        # Index doesn't exist yet?
        return {}
    if version != self._version:
        # This is another version.  Avoid trying to unpickling the
        # rest of the stream, as that may fail.
        return {}
    stamp, overloads = pickle.loads(data)
    if stamp != self._source_stamp:
        print("Stamp not equal")  # noqa: T001, T201
        print(f"Stamp: {stamp}")  # noqa: T001, T201
        print(f"Source stamp: {self._source_stamp}")  # noqa: T001, T201

    original_result = original_load_index(self)  # Calling the original method
    data = f"Cache loaded with the following data: {original_result}"
    print(data)  # noqa: T001, T201
    return original_result


original_load_index = numba.core.caching.IndexDataCacheFile._load_index

numba.core.caching.IndexDataCacheFile._load_index = custom_load_index


@pytest.mark.parametrize("dist", DISTANCES)
def test_numba_cache(dist):
    print("CALLED")  # noqa: T001, T201
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    dist["distance"](x, y)
