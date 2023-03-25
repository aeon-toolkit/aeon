# -*- coding: utf-8 -*-

"""aeon."""

__version__ = "0.1.0rc0"

__all__ = ["show_versions", "get_config", "set_config", "config_context"]

from aeon._config import config_context, get_config, set_config
from aeon.utils._maint._show_versions import show_versions
