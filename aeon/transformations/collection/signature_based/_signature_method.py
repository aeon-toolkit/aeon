"""Signature transformer."""

from sklearn.pipeline import Pipeline

from aeon.transformations.collection import BaseCollectionTransformer
from aeon.transformations.collection.signature_based._augmentations import (
    _make_augmentation_pipeline,
)
from aeon.transformations.collection.signature_based._compute import (
    _WindowSignatureTransform,
)


class SignatureTransformer(BaseCollectionTransformer):
    """Transformation class from the signature method.

    Follows the methodology laid out in the paper:
        "A Generalised Signature Method for Multivariate Time Series"

    Parameters
    ----------
    augmentation_list: tuple of strings, contains the augmentations to be
        applied before application of the signature transform.
    window_name: str, The name of the window transform to apply.
    window_depth: int, The depth of the dyadic window. (Active only if
        `window_name == 'dyadic'`).
    window_length: int, The length of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding']`.
    window_step: int, The step of the sliding/expanding window. (Active
        only if `window_name in ['sliding, 'expanding']`.
    rescaling: str or None, The method of signature rescaling.
    sig_tfm: str, String to specify the type of signature transform. One of:
        ['signature', 'logsignature']).
    depth: int, Signature truncation depth.

    Attributes
    ----------
    signature_method: sklearn.Pipeline, A sklearn pipeline object that contains
        all the steps to extract the signature features.
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "python_dependencies": "esig",
        "python_version": "<3.11",
    }

    def __init__(
        self,
        augmentation_list=("basepoint", "addtime"),
        window_name="dyadic",
        window_depth=3,
        window_length=None,
        window_step=None,
        rescaling=None,
        sig_tfm="signature",
        depth=4,
    ):
        self.augmentation_list = augmentation_list
        self.window_name = window_name
        self.window_depth = window_depth
        self.window_length = window_length
        self.window_step = window_step
        self.rescaling = rescaling
        self.sig_tfm = sig_tfm
        self.depth = depth

        super().__init__()
        self.setup_feature_pipeline()

    def setup_feature_pipeline(self):
        """Set up the signature method as an sklearn pipeline."""
        augmentation_step = _make_augmentation_pipeline(self.augmentation_list)
        transform_step = _WindowSignatureTransform(
            window_name=self.window_name,
            window_depth=self.window_depth,
            window_length=self.window_length,
            window_step=self.window_step,
            sig_tfm=self.sig_tfm,
            sig_depth=self.depth,
            rescaling=self.rescaling,
        )

        # The so-called 'signature method' as defined in the reference paper
        self.signature_method = Pipeline(
            [
                ("augmentations", augmentation_step),
                ("window_and_transform", transform_step),
            ]
        )

    def _fit(self, X, y=None):
        self.signature_method.fit(X)
        return self

    def _transform(self, X, y=None):
        return self.signature_method.transform(X)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        params = {
            "augmentation_list": ("basepoint", "addtime"),
            "depth": 3,
            "window_name": "global",
        }
        return params
