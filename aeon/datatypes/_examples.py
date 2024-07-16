"""Example fixtures for mtypes/scitypes.

Exports
-------
get_examples(mtype: str, as_scitype: str, return_lossy=False)
    retrieves examples for mtype/scitype, and/or whether it is lossy

examples[i] are assumed "content-wise the same" for the same as_scitype, i
    only in different machine representations

the representation is considered "lossy" if the representation is incomplete
    e.g., metadata such as column names are missing
"""

from aeon.datatypes._registry import mtype_to_scitype

__maintainer__ = []

__all__ = [
    "get_examples",
]


from aeon.datatypes._hierarchical import (
    example_dict_Hierarchical,
    example_dict_lossy_Hierarchical,
    example_dict_metadata_Hierarchical,
)
from aeon.datatypes._panel import (
    example_dict_lossy_Panel,
    example_dict_metadata_Panel,
    example_dict_Panel,
)
from aeon.datatypes._proba import (
    example_dict_lossy_Proba,
    example_dict_metadata_Proba,
    example_dict_Proba,
)
from aeon.datatypes._series import (
    example_dict_lossy_Series,
    example_dict_metadata_Series,
    example_dict_Series,
)
from aeon.datatypes._table import (
    example_dict_lossy_Table,
    example_dict_metadata_Table,
    example_dict_Table,
)

# pool example_dict-s
example_dict = dict()
example_dict.update(example_dict_Series)
example_dict.update(example_dict_Panel)
example_dict.update(example_dict_Hierarchical)
example_dict.update(example_dict_Table)
example_dict.update(example_dict_Proba)

example_dict_lossy = dict()
example_dict_lossy.update(example_dict_lossy_Series)
example_dict_lossy.update(example_dict_lossy_Panel)
example_dict_lossy.update(example_dict_lossy_Hierarchical)
example_dict_lossy.update(example_dict_lossy_Table)
example_dict_lossy.update(example_dict_lossy_Proba)

example_dict_metadata = dict()
example_dict_metadata.update(example_dict_metadata_Series)
example_dict_metadata.update(example_dict_metadata_Panel)
example_dict_metadata.update(example_dict_metadata_Hierarchical)
example_dict_metadata.update(example_dict_metadata_Table)
example_dict_metadata.update(example_dict_metadata_Proba)


def get_examples(
    mtype: str,
    as_scitype: str = None,
    return_lossy: bool = False,
    return_metadata: bool = False,
):
    """Retrieve a dict of examples for mtype `mtype`, scitype `as_scitype`.

    Parameters
    ----------
    mtype: str - name of the mtype for the example, a valid mtype string
        valid mtype strings, with explanation, are in datatypes.TYPE_REGISTER
    as_scitype : str, optional - name of scitype of the example, a valid scitype string
        valid scitype strings, with explanation, are in datatypes.DATATYPE_REGISTER
        default = inferred from mtype of obj
    return_lossy: bool, optional, default=False
        whether to return second argument
    return_metadata: bool, optional, default=False
        whether to return third argument

    Returns
    -------
    fixtures: dict with integer keys, elements being
        fixture - example for mtype `mtype`, scitype `as_scitype`
        if return_lossy=True, elements are pairs with fixture and
            lossy: bool - whether the example is a lossy representation
        if return_metadata=True, elements are triples with fixture, lossy, and
            metadata: dict - metadata dict that would be returned by check_is_mtype
    """
    # if as_scitype is None, infer from mtype
    if as_scitype is None:
        as_scitype = mtype_to_scitype(mtype)

    # retrieve all keys that match the query
    exkeys = example_dict.keys()
    keys = [k for k in exkeys if k[0] == mtype and k[1] == as_scitype]

    # retrieve all fixtures that match the key
    fixtures = dict()

    for k in keys:
        if return_lossy:
            fixtures[k[2]] = (example_dict.get(k), example_dict_lossy.get(k))
        elif return_metadata:
            fixtures[k[2]] = (
                example_dict.get(k),
                example_dict_lossy.get(k),
                example_dict_metadata.get((k[1], k[2])),
            )
        else:
            fixtures[k[2]] = example_dict.get(k)

    return fixtures
