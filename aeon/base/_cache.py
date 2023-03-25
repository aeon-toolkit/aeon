# -*- coding: utf-8 -*-
from functools import wraps

import joblib
from cachetools import LRUCache

from aeon import get_config


def cache(func):
    """Decorate functions to cache return."""
    if get_config()["cache"]:
        cache = LRUCache(maxsize=1000)

        @wraps(func)
        def cached_func(*args, **kwargs):
            # Convert unhashable objects to hashable ones
            key = joblib.hash((args, kwargs))
            if key in cache:
                return cache[key]
            else:
                result = func(*args, **kwargs)
                cache[key] = result
                return result

        return cached_func
    else:
        return func
