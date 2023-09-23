.. _developer_guide_deprecation:

===========
Deprecation
===========

Please note this policy is not currently in force. We aim to restore deprecation for
V0.5 but the policy may be revised, but it will be based on the below.

Deprecation policy
==================

aeon `releases <https://github.com/aeon-toolkit/aeon/releases>`_ follow `semantic versioning <https://semver.org>`_.
A release number denotes <major>.<minor>.<patch> versions.

Our previous deprecation policy is as follows:

* all interface breaking (not downwards compatible) changes to public interfaces must
be accompanied by deprecation.
* such changes or removals happen only at MINOR or MAJOR versions, not at PATCH versions.
* deprecation warnings should be included for at least one MINOR version cycle
before change or removal.


Deprecation process
===================

The deprecation process is as follows:

* **Raise a warning.** For all deprecated functionality, we raise a
:code:`FutureWarning`. The warning message should give the version number when the functionality will be changed and describe the new usage.
* **Add a TODO comment in the code.** Add the following TODO comment to all pieces of code that can be removed: :code:`TODO: remove in <version-number>`. For example, :code:`TODO: remove in v0.11.0`. The TODO comment should describe which steps need to be taken for deprecation (e.g. removal of arguments, removal of functions or blocks of code). If changes need to be applied across multiple places, place multiple TODO comments.
* **Remove deprecated functionality.** As part of the release process, all deprecated functionality that is due to be removed will be removed by searching for the TODO comments.


To raise the warning, we use the `deprecated <https://deprecated.readthedocs.io/en/latest/index.html>`_ package.
The package provides depreciation helper functions such as the :code:`deprecated` decorator.
When importing it from :code:`deprecated.sphinx`, it automatically adds a deprecation message to the docstring.
You can decorate functions, methods or classes.

Examples
--------

In the examples below, the :code:`deprecated` decorator will raise a :code:`FutureWarning` saying that the functionality has been deprecated since version 0.9.0 and will be removed in version 0.11.0.

Functions
~~~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.7.0
    @deprecated(version="0.6.0", reason="my_old_function will be removed in v0.7.0",
category=FutureWarning)
    def my_old_function(x, y):
        return x + y

Methods
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        # TODO: remove in v0.7.0
        @deprecated(version="0..0", reason="my_old_method will be removed in v0.7.0",
category=FutureWarning)
        def my_old_method(self, x, y):
            return x + y

Classes
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.7.0
    @deprecated(version="0.6.0", reason="MyOldClass will be removed in v0.7.0",
category=FutureWarning)
    class MyOldClass:
        pass


Special deprecations
====================

This section outlines the deprecation process for cases which use of ``deprecated`` does not cover.

Deprecating tags
----------------

To deprecate tags, it needs to be ensured that warnings are raised when the tag is used.
There are two common scenarios: removing a tag, or renaming a tag.

For either scenario, the helper class ``TagAliaserMixin`` (in ``aeon.base``) can be used.

To deprecate tags, add the ``TagAliaserMixin`` to ``BaseEstimator``, or another ``BaseObject`` descendant.
It is advised to select the youngest descendant that fully covers use of the deprecated tag.
``TagAliaserMixin`` overrides the tag family of methods, and should hence be the first class to inherit from
(or in case of multiple mixins, earlier than ``BaseObject``).

``alias_dict`` in ``TagAliaserMixin`` contains a dictionary of deprecated tags:
For removal, add an entry ``"old_tag_name": ""``.
For renaming, add an entry ``"old_tag_name": "new_tag_name"``
``deprecate_dict`` contains the version number of renaming or removal, and should have the same keys as ``alias_dict``.

The ``TagAliaserMixin`` class will ensure that new tags alias old tags and vice versa, during
the deprecation period. Informative warnings will be raised whenever the deprecated tags are being accessed.

When removing/renaming tags after the deprecation period,
ensure to remove the removed tags from the dictionaries in ``TagAliaserMixin`` class.
If no tags are deprecated anymore (e.g., all deprecated tags are removed/renamed),
ensure to remove this class as a parent of ``BaseObject`` or ``BaseEstimator``.
