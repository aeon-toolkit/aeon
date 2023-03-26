# -*- coding: utf-8 -*-
from functools import wraps

import joblib
from cachetools import LRUCache

from aeon import get_config, set_config


def cache(func):
    """Decorate functions to cache return."""

    @wraps(func)
    def cached_func(*args, **kwargs):
        if get_config()["cache"]:
            # Convert unhashable objects to hashable ones
            key = joblib.hash((args, kwargs))
            if get_config()["_cache"] is None:
                _cache = LRUCache(maxsize=1)
                # _cache = {}
            else:
                _cache = get_config()["_cache"]
            if key in _cache:
                return _cache[key]
            else:
                result = func(*args, **kwargs)
                _cache[key] = result
                set_config(_cache=_cache)

                return result
        else:
            return func(*args, **kwargs)

    return cached_func
