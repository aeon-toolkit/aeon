"""Utility class for deprecating tags."""

import warnings
from copy import deepcopy


class _TagAliaserMixin:
    """
    Mixin class for tag aliasing and deprecation of old tags.

    To deprecate tags, add the TagAliaserMixin to BaseObject or BaseEstimator.
    alias_dict contains the deprecated tags, and supports removal and renaming.
        For removal, add an entry "old_tag_name": ""
        For renaming, add an entry "old_tag_name": "new_tag_name"
    deprecate_dict contains the version number of renaming or removal.
        the keys in deprecate_dict should be the same as in alias_dict.
        values in deprecate_dict should be strings, the version of removal/renaming.

    The class will ensure that new tags alias old tags and vice versa, during
    the deprecation period. Informative warnings will be raised whenever the
    deprecated tags are being accessed.

    When removing tags, ensure to remove the removed tags from this class.
    If no tags are deprecated anymore (e.g., all deprecated tags are removed/renamed),
    ensure toremove this class as a parent of BaseObject or BaseEstimator.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def get_class_tags(cls):
        """
        Get class tags from estimator class and all its parent classes.

        Get the tags relating to the class not a particular object.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from _tags
            class attribute via nested inheritance. NOT overridden by dynamic
            tags set by set_tags or mirror_tags.

        See Also
        --------
        get_tag : Get a single tag from an object.
        get_clas_tags : Get all tags from a class.
        get_tags : Get all tags from an object.

        Examples
        --------
        >>> from aeon.classification import DummyClassifier
        >>> tags = DummyClassifier.get_class_tags()
        """
        collected_tags = super().get_class_tags()
        collected_tags = cls._complete_dict(collected_tags)
        return collected_tags

    @classmethod
    def get_class_tag(cls, tag_name, tag_value_default=None):
        """
        Get tag value from estimator class (only class tags).

        Parameters
        ----------
        tag_name : str
            Name of tag value.
        tag_value_default : any type
            Default/fallback value if tag is not found.

        Returns
        -------
        tag_value :
            Value of the `tag_name` tag in self. If not found, returns
            `tag_value_default`.

        See Also
        --------
        get_tag : Get a single tag from an object.
        get_clas_tags : Get all tags from a class.
        get_tags : Get all tags from an object.

        Examples
        --------
        >>> from aeon.classification import DummyClassifier
        >>> DummyClassifier.get_class_tag("capability:multivariate")
        True
        """
        cls._deprecate_tag_warn([tag_name])
        return super().get_class_tag(
            tag_name=tag_name, tag_value_default=tag_value_default
        )

    def get_tags(self):
        """
        Get tags from estimator class and dynamic tag overrides.

        Returns
        -------
        collected_tags : dict
            Dictionary of tag name : tag value pairs. Collected from _tags
            class attribute via nested inheritance and then any overrides
            and new tags from _tags_dynamic object attribute.
        """
        collected_tags = super().get_tags()
        collected_tags = self._complete_dict(collected_tags)
        return collected_tags

    def get_tag(self, tag_name, tag_value_default=None, raise_error=True):
        """
        Get tag value from estimator class and dynamic tag overrides.

        Parameters
        ----------
        tag_name : str
            Name of tag to be retrieved.
        tag_value_default : any type, default=None
            Default/fallback value if tag is not found.
        raise_error : bool
            whether a ValueError is raised when the tag is not found.

        Returns
        -------
        string or None
            Value of the `tag_name` tag in self. If not found, returns an error if
            raise_error is True, otherwise it returns `tag_value_default`.

        Raises
        ------
        ValueError if raise_error is True i.e. if tag_name is not in self.get_tags(
        ).keys()
        """
        self._deprecate_tag_warn([tag_name])
        return super().get_tag(
            tag_name=tag_name,
            tag_value_default=tag_value_default,
            raise_error=raise_error,
        )

    def set_tags(self, **tag_dict):
        """
        Set dynamic tags to given values.

        Parameters
        ----------
        tag_dict : dict
            Dictionary of tag name : tag value pairs.

        Returns
        -------
        Self :
            Reference to self.

        Notes
        -----
        Changes object state by settting tag values in tag_dict as dynamic tags
        in self.
        """
        self._deprecate_tag_warn(tag_dict.keys())

        tag_dict = self._complete_dict(tag_dict)
        super().set_tags(**tag_dict)
        return self

    @classmethod
    def _complete_dict(cls, tag_dict):
        """Add all aliased and aliasing tags to the dictionary."""
        alias_dict = cls.alias_dict
        deprecated_tags = set(tag_dict.keys()).intersection(alias_dict.keys())
        new_tags = set(tag_dict.keys()).intersection(alias_dict.values())

        if len(deprecated_tags) > 0 or len(new_tags) > 0:
            new_tag_dict = deepcopy(tag_dict)
            # for all tag strings being set, write the value
            #   to all tags that could *be aliased by* the string
            #   and all tags that could be *aliasing* the string
            # this way we ensure upwards and downwards compatibility
            for old_tag, new_tag in alias_dict.items():
                for tag in tag_dict:
                    if tag == old_tag and new_tag != "":
                        new_tag_dict[new_tag] = tag_dict[tag]
                    if tag == new_tag:
                        new_tag_dict[old_tag] = tag_dict[tag]
            return new_tag_dict
        else:
            return tag_dict

    @classmethod
    def _deprecate_tag_warn(cls, tags):
        """
        Print warning message for tag deprecation.

        Parameters
        ----------
        tags : list of str.

        Raises
        ------
        DeprecationWarning for each tag in tags that is aliased by cls.alias_dict.
        """
        for tag_name in tags:
            if tag_name in cls.alias_dict.keys():
                version = cls.deprecate_dict[tag_name]
                new_tag = cls.alias_dict[tag_name]
                msg = f"tag {tag_name!r} will be removed in aeon version {version}"
                if new_tag != "":
                    msg += (
                        f" and replaced by {new_tag!r}, please use {new_tag!r} instead"
                    )
                else:
                    msg += ', please remove code that access or sets "{tag_name}"'
                warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
