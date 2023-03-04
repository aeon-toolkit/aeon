# -*- coding: utf-8 -*-
"""Global configuration state and functions for management."""
import threading
from contextlib import contextmanager as contextmanager

_global_config = {
    "check_X_y": False,
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration.

    If the configuration does not exist, copy the default global configuration.
    """
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global sktime configuration.
    set_config : Set global sktime configuration.
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(
    check_X_y=None,
):
    """Set global sktime configuration.

    Parameters
    ----------
    check_X_y: bool, default=None

    See Also
    --------
    config_context : Context manager for global sktime configuration.
    get_config : Retrieve current values of the global configuration.
    """
    local_config = _get_threadlocal_config()

    if check_X_y is not None:
        local_config["check_X_y"] = check_X_y


@contextmanager
def config_context(
    *,
    check_X_y=None,
):
    """Context manager for global sktime configuration.

    Parameters
    ----------
    check_X_y: bool, default=None
        If True, will make checks on X and y

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global sktime configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.
    """
    old_config = get_config()
    set_config(
        check_X_y=check_X_y,
    )

    try:
        yield
    finally:
        set_config(**old_config)
