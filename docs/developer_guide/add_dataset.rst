====================
Adding a New Dataset
====================

To add a new dataset into :code:`aeon` internal dataset repository, please proceed with the following steps:

1. From the root of your :code:`aeon` local repository, create a :code:`<dataset-name>` folder:

   .. code-block:: shell

      mkdir ./datasets/data/<dataset-name>

2. In the above directory, add your dataset file :code:`<dataset-name>.<EXT>`, where :code:`<EXT>` is the file extension:

   * The list of supported file formats is available in the :code:`aeon/MANIFEST.in` file (*e.g.*, :code:`.csv`, :code:`.txt`).
   * If your file format ``<EXT>`` does not figure in the list, simply add it in the :code:`aeon/MANIFEST.in` file:
   ::

      "aeon/MANIFEST.in"
      ...
      recursive-include aeon/datasets *.csv ... *.<EXT>
      ...

3. In ``aeon/datasets/_single_problem_loaders.py``, declare a :code:`load_<dataset-name>(...)` function. Feel free to use any other declared functions as templates for either classification or regression datasets.

4. In ``aeon/datasets/__init__.py``, append :code:`"load_<dataset-name>"` to the list :code:`__all__`.

5. In ``aeon/datasets/setup.py``, append :code:`"<dataset-name>"` to the tuple :code:`included_datasets`.
