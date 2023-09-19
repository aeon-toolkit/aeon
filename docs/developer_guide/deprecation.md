.. _developer_guide_deprecation:

===========
Deprecation
===========

Deprecation policy
==================

aeon `releases <https://github.com/aeon-toolkit/aeon/releases>`_ follow `semantic versioning <https://semver.org>`_.
A release number denotes <major>.<minor>.<patch> versions.

Broadly, if a change could unexpectedly cause code using aeon to crash, then it
should be deprecated to give the user a chance to prepare.

- When to deprecate:
    - Removal or renaming of public classes or functions;
    - Removal or renaming of public class or function parameters;
    - Addition of positional parameters without default values.

Deprecation warnings should be included for at least one minor version cycle before
change or removal.

Note that the deprecation policy does not necessarily apply to modules we class as 
still experimental. Currently experimental modules are

- annotation

When we introduce a new module, we may classify it as experimental until the API is 
stable. We will try not make drastic changes to experimental modules, but we need to 
retain the freedom to be more agile with the design in these cases.  

Deprecation process
===================

To deprecate, write a :code:`TODO` comment stating the version the code should be
removed in and raise a warning using use the `deprecated <https://deprecated
.readthedocs.io/en/latest/index.html>`_ package. This raises  a :code:`FutureWarning`
saying that the functionality has been deprecated. Import from :code:`deprecated
.sphinx` so the deprecation message is automatically added to the docstring.

Examples
--------

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.7.0
    @deprecated(version="0.6.0", reason="my_old_function will be removed in v0.7.0",
category=FutureWarning)
    def my_old_function(x, y):
        return x + y

    # TODO: remove in v0.7.0
    @deprecated(version="0.6.0", reason="my_function will be have a third positional
parameter z in v0.7.0",
category=FutureWarning)
    def my_function(x, y):
        return x + y


.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        # TODO: remove in v0.7.0
        @deprecated(version="0.6.0", reason="my_method parameter x will be removed in
v0.7.0", category=FutureWarning)
        def my_method(self, x, y):
            return x + y

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.7.0
    @deprecated(version="0.6.0", reason="MyOldClass will be renamed MyNewClass in v0.7.0",
category=FutureWarning)
    class MyOldClass:
        pass
