"""Generate test examples."""

import numpy as np
import pandas as pd

# pd.Series
s1 = pd.Series([1, 4, 0.5, -3], dtype=np.float64, name="a")
series_examples = [s1]
# pd.DataFrame univariate and multivariate
d1 = pd.DataFrame({"a": [1, 4, 0.5, -3]})
d2 = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})
dataframe_examples = [d1, d2]
# pd-multiindex multivariate, equally sampled
cols = ["instances", "timepoints"] + ["var_0", "var_1"]
mi1list = [
    pd.DataFrame([[0, 0, 1, 4], [0, 1, 2, 5], [0, 2, 3, 6]], columns=cols),
    pd.DataFrame([[1, 0, 1, 4], [1, 1, 2, 55], [1, 2, 3, 6]], columns=cols),
    pd.DataFrame([[2, 0, 1, 42], [2, 1, 2, 5], [2, 2, 3, 6]], columns=cols),
]
mi1 = pd.concat(mi1list)
mi1 = mi1.set_index(["instances", "timepoints"])

cols = ["instances", "timepoints"] + ["var_0"]
mi2list = [
    pd.DataFrame([[0, 0, 4], [0, 1, 5], [0, 2, 6]], columns=cols),
    pd.DataFrame([[1, 0, 4], [1, 1, 55], [1, 2, 6]], columns=cols),
    pd.DataFrame([[2, 0, 42], [2, 1, 5], [2, 2, 6]], columns=cols),
]
mi2 = pd.concat(mi2list)
mi2 = mi2.set_index(["instances", "timepoints"])
multiindex_examples = [mi1, mi2]

# pd_multiindex_hier
cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(2)]
mih1list = [
    pd.DataFrame(
        [["a", 0, 0, 1, 4], ["a", 0, 1, 2, 5], ["a", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 1, 0, 1, 4], ["a", 1, 1, 2, 55], ["a", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["a", 2, 0, 1, 42], ["a", 2, 1, 2, 5], ["a", 2, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 0, 0, 1, 4], ["b", 0, 1, 2, 5], ["b", 0, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 1, 0, 1, 4], ["b", 1, 1, 2, 55], ["b", 1, 2, 3, 6]], columns=cols
    ),
    pd.DataFrame(
        [["b", 2, 0, 1, 42], ["b", 2, 1, 2, 5], ["b", 2, 2, 3, 6]], columns=cols
    ),
]
mih1 = pd.concat(mih1list)
mih1 = mih1.set_index(["foo", "bar", "timepoints"])

cols = ["foo", "bar", "timepoints"] + [f"var_{i}" for i in range(1)]

mih2list = [
    pd.DataFrame([["a", 0, 0, 1], ["a", 0, 1, 2], ["a", 0, 2, 3]], columns=cols),
    pd.DataFrame([["a", 1, 0, 1], ["a", 1, 1, 2], ["a", 1, 2, 3]], columns=cols),
    pd.DataFrame([["a", 2, 0, 1], ["a", 2, 1, 2], ["a", 2, 2, 3]], columns=cols),
    pd.DataFrame([["b", 0, 0, 1], ["b", 0, 1, 2], ["b", 0, 2, 3]], columns=cols),
    pd.DataFrame([["b", 1, 0, 1], ["b", 1, 1, 2], ["b", 1, 2, 3]], columns=cols),
    pd.DataFrame([["b", 2, 0, 1], ["b", 2, 1, 2], ["b", 2, 2, 3]], columns=cols),
]
mih2 = pd.concat(mih2list)
mih2 = mih2.set_index(["foo", "bar", "timepoints"])
mih_examples = [mih1, mih2]

np1 = np.array([1, 2, 3, 4, 5])
np2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
np_examples = [np1, np2]


def get_hierarchical_examples():
    """Get hierarchical for tests."""
    return mih_examples


def get_series_examples():
    """Get single series examples."""
    return [series_examples, dataframe_examples]


def get_collection_examples():
    """Get example collections."""
    return [mih1]


def get_examples(datatype: str):
    """Create two examples of each possible type."""
    if datatype == "pd.Series":
        return series_examples
    elif datatype == "pd.DataFrame":
        return dataframe_examples
    elif datatype == "pd-multiindex":
        return multiindex_examples
    elif datatype == "pd_multiindex_hier":
        return mih_examples
    elif datatype == "np.ndarray":
        return np_examples
    else:
        raise ValueError(f"Unknown datatype : {datatype} in get examples.")
