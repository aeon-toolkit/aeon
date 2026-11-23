"""Shared testing utilities for LAPACK errors."""

LAPACK_ERROR = (
    "Indexing a matrix of 3365692186 elements would incur an integer overflow in "
    "LAPACK. Try using numpy.linalg.svd instead."
)


def _failing_fit(error):
    """Return a replacement ``fit`` method that raises ``error``."""

    def fit(self, *args, **kwargs):
        raise error

    return fit
