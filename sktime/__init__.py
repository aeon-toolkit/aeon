# -*- coding: utf-8 -*-

"""sktime."""

__version__ = "0.16.0"

__all__ = ["show_versions", "get_config", "set_config", "config_context"]

from sktime._config import config_context, get_config, set_config
from sktime.utils._maint._show_versions import show_versions
