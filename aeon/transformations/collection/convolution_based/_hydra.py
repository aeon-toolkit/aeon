from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs


class HydraTransformer(BaseCollectionTransformer):
    """Hydra Transformer."""

    _tags = {
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
        "fit_is_empty": True,
    }

    def __init__(self, k=8, g=64, n_jobs=1, random_state=None):
        self.k = k
        self.g = g
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _transform(self, X, y=None):
        import torch

        from aeon.transformations.collection.convolution_based._torch._hydra_torch import (  # noqa: E501
            _HydraInternal,
        )

        if isinstance(self.random_state, int):
            torch.manual_seed(self.random_state)

        n_jobs = check_n_jobs(self.n_jobs)
        torch.set_num_threads(n_jobs)

        self.hydra = _HydraInternal(X.shape[-1], k=self.k, g=self.g)
        return self.hydra(torch.tensor(X).float())
