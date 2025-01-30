"""Implements meta estimator for estimators composed of other estimators."""

__maintainer__ = ["MatthewMiddlehurst"]
__all__ = ["ComposableEstimatorMixin"]

from abc import ABC, abstractmethod

from aeon.base import BaseAeonEstimator
from aeon.base._base import _clone_estimator


class ComposableEstimatorMixin(ABC):
    """Handles parameter management for estimators composed of named estimators.

    Parts (i.e. get_params and set_params) adapted or copied from the scikit-learn
    ``_BaseComposition`` class in utils/metaestimators.py.
    """

    # Attribute name containing an iterable of processed (str, estimator) tuples
    # with unfitted estimators and unique names. Used in get_params and set_params
    _estimators_attr = "_estimators"
    # Attribute name containing an iterable of fitted (str, estimator) tuples.
    # Used in get_fitted_params
    _fitted_estimators_attr = "estimators_"

    @abstractmethod
    def __init__(self):
        super().__init__()

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Returns the parameters given in the constructor as well as the
        estimators contained within the composable estimator if deep.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = super().get_params(deep=deep)
        if not deep:
            return out

        estimators = getattr(self, self._estimators_attr)
        out.update(estimators)

        for name, estimator in estimators:
            for key, value in estimator.get_params(deep=True).items():
                out[f"{name}__{key}"] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained composable
        estimator using their assigned name.

        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            within the composable estimator. Parameters of the estimators may be set
            using its name and the parameter name separated by a '__'.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        # Ensure strict ordering of parameter setting:
        # 1. All steps
        if self._estimators_attr in params:
            setattr(self, self._estimators_attr, params.pop(self._estimators_attr))

        # 2. Replace items with estimators in params
        items = getattr(self, self._estimators_attr)
        if isinstance(items, list) and items:
            # Get item names used to identify valid names in params
            item_names, _ = zip(*items)
            for name in list(params.keys()):
                if "__" not in name and name in item_names:
                    self._replace_estimator(
                        self._estimators_attr, name, params.pop(name)
                    )

        # 3. Step parameters and other initialisation arguments
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        # assumes `name` is a valid estimator name
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)

    def get_fitted_params(self, deep=True):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the fitted parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        fitted_params : dict
            Fitted parameter names mapped to their values.
        """
        self._check_is_fitted()

        out = super().get_fitted_params(deep=deep)
        if not deep:
            return out

        estimators = getattr(self, self._fitted_estimators_attr)
        out.update(estimators)

        for name, estimator in estimators:
            for key, value in self._get_fitted_params(estimator, deep=True).items():
                out[f"{name}__{key}"] = value
        return out

    def _check_estimators(
        self,
        estimators,
        attr_name="estimators",
        class_type=BaseAeonEstimator,
        allow_tuples=True,
        allow_single_estimators=True,
        unique_names=True,
        invalid_names=None,
    ):
        """Check that estimators is a list of estimators or list of str/est tuples.

        Parameters
        ----------
        estimators : list
            A list of estimators or list of (str, estimator) tuples.
        attr_name : str, optional. Default = "steps"
            Name of checked attribute in error messages
        class_type : class, tuple of class or None, default=BaseAeonEstimator.
            Class(es) that all estimators in ``estimators`` are checked to be an
            instance of.
        allow_tuples : boolean, default=True.
            Whether tuples of (str, estimator) are allowed in ``estimators``.
            Generally, the end-state we want is a list of tuples, so this should be True
            in most cases.
        allow_single_estimators : boolean, default=True.
            Whether non-tuple estimator classes are allowed in ``estimators``.
        unique_names : boolean, default=True.
            Whether to check that all tuple strings in `estimators` are unique.
        invalid_names : str, list of str or None, default=None.
            Names that are invalid for estimators in ``estimators``.

        Raises
        ------
        TypeError
            If estimators not valid for the given configuration.
        """
        if (
            estimators is None
            or len(estimators) == 0
            or not isinstance(estimators, list)
        ):
            raise TypeError(
                f"Invalid {attr_name} attribute, {attr_name} should be a list."
            )

        if invalid_names is not None and isinstance(invalid_names, str):
            invalid_names = [invalid_names]

        param_names = self.get_params(deep=False).keys()
        names = []
        for obj in estimators:
            if isinstance(obj, tuple):
                if not allow_tuples:
                    raise ValueError(
                        f"{attr_name} should only contain singular estimators instead "
                        f"of (str, estimator) tuples."
                    )
                if not len(obj) == 2 or not isinstance(obj[0], str):
                    raise ValueError(
                        f"All tuples in {attr_name} must be of form (str, estimator)."
                    )
                if not isinstance(obj[1], class_type):
                    raise ValueError(
                        f"All estimators in {attr_name} must be an instance "
                        f"of {class_type}."
                    )
                if obj[0] in param_names:
                    raise ValueError(
                        f"Estimator name conflicts with constructor arguments: {obj[0]}"
                    )
                if "__" in obj[0]:
                    raise ValueError(f"Estimator name must not contain __: {obj[0]}")
                if invalid_names is not None and obj[0] in invalid_names:
                    raise ValueError(f"Estimator name is invalid: {obj[0]}")
                if unique_names:
                    if obj[0] in names:
                        raise ValueError(
                            f"Names in {attr_name} must be unique. Found duplicate "
                            f"name: {obj[0]}."
                        )
                    else:
                        names.append(obj[0])
            elif isinstance(obj, class_type):
                if not allow_single_estimators:
                    raise ValueError(
                        f"{attr_name} should only contain (str, estimator) tuples "
                        f"instead of singular estimators."
                    )
            else:
                raise TypeError(
                    f"All elements in {attr_name} must be a (str, estimator) tuple or "
                    f"estimator type of {class_type}."
                )

    def _convert_estimators(self, estimators, clone_estimators=True):
        """Convert estimators to list of (str, estimator) tuples.

        Assumes ``_check_estimators`` has already been called on ``estimators``.

        Parameters
        ----------
        estimators : list of estimators, or list of (str, estimator tuples)
            A list of estimators or list of (str, estimator) tuples to be converted.
        clone_estimators : boolean, default=True.
            Whether to return clone of estimators in ``estimators`` (True) or
            references (False).

        Returns
        -------
        estimator_tuples : list of (str, estimator) tuples
            If estimators was a list of (str, estimator) tuples, then identical/cloned
            to ``estimators``.
            if was a list of estimators or mixed, then unique str are generated to
            create tuples.
        """
        cloned_ests = []
        names = []
        name_dict = {}
        for est in estimators:
            if isinstance(est, tuple):
                name = est[0]
                cloned_ests.append(
                    _clone_estimator(est[1]) if clone_estimators else est[1]
                )
            else:
                name = est.__class__.__name__
                cloned_ests.append(_clone_estimator(est) if clone_estimators else est)

            if name not in name_dict and name in names:
                name_dict[name] = 0
            names.append(name)

        estimator_tuples = []
        for i, est in enumerate(cloned_ests):
            if names[i] in name_dict:
                estimator_tuples.append((f"{names[i]}_{name_dict[names[i]]}", est))
                name_dict[names[i]] += 1
            else:
                estimator_tuples.append((names[i], est))

        return estimator_tuples
