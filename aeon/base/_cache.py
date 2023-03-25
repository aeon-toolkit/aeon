# -*- coding: utf-8 -*-
from functools import wraps

import numpy as np
from cachetools import LRUCache

from aeon import get_config


def make_hashable(obj):
    """Convert an unhashable object to a hashable one."""
    # Replace unhashable types with their string representation
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()
    else:
        return str(obj)


def cache(func):
    """Decorate functions to cache return."""
    if get_config()["cache"]:
        cache = LRUCache(maxsize=1000)

        @wraps(func)
        def cached_func(*args, **kwargs):
            # Convert unhashable objects to hashable ones
            key = make_hashable((args, kwargs))
            if key in cache:
                return cache[key]
            else:
                result = func(*args, **kwargs)
                cache[key] = result
                return result

        return cached_func
    else:
        return func
