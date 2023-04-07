
.. _user_guide:

==========
User Guide
==========

Welcome to aeon's user guide!

The user guide consists of introductory notebooks, ordered by learning task.

For guided tutorials with videos, see our :ref:`tutorials` page.

To run the user guide notebooks interactively, you can
`launch them on binder <https://mybinder.org/v2/gh/aeon-toolkit/aeon/main?filepath=examples>`_
without having to install anything.

We assume basic familiarity with `scikit-learn`_. If you haven’t worked with scikit-learn before, check out their
`getting-started guide`_.

The notebook files can be found `here <https://github.com/aeon-toolkit/aeon/blob/main/examples>`_.

.. _scikit-learn: https://scikit-learn.org/stable/
.. _getting-started guide: https://scikit-learn.org/stable/getting_started.html

.. grid:: 2 4 4 4
    :gutter: 1

    .. grid-item-card::
        :img-top: examples/img/forecasting2.png
        :link: /examples/01_forecasting.ipynb
        :link-type: ref
        :text-align: center

        Forecasting with sktime

    .. grid-item-card::
        :link: /examples/01a_forecasting_sklearn.ipynb
        :link-type: ref
        :text-align: center

        Forecasting with sktime - appendix: forecasting, supervised regression, and pitfalls in confusing the two


    .. grid-item-card::
        :link: /examples/01b_forecasting_proba.ipynb
        :link-type: ref
        :text-align: center

        Probabilistic Forecasting with sktime

    .. grid-item-card::
        :img-top: examples/img/tsc.png
        :link: /examples/classification/classification.ipynb
        :link-type: ref
        :text-align: center

        Time Series Classification with aeon

    .. grid-item-card::
        :link: /examples/04_benchmarking.ipynb
        :link-type: ref
        :text-align: center

        Benchmarking with sktime

    .. grid-item-card::
        :link: /examples/AA_datatypes_and_datasets.ipynb
        :link-type: ref
        :text-align: center

        In-memory data representations and data loading
