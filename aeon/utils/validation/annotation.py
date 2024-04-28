"""Series of checks for annotation classes."""

__maintainer__ = ["TonyBagnall"]

__all__ = ["check_fmt", "check_labels"]


def check_fmt(fmt):
    """Check annotation format.

    Parameters
    ----------
    fmt : str {"sparse", "dense"}
        annotation format

    Returns
    -------
    fmt : str
        Checked annotation format.
    """
    valid_fmts = ["sparse", "dense"]
    if fmt not in valid_fmts:
        raise ValueError(f"`fmt` must be in: {valid_fmts}, but found: {fmt}.")


def check_labels(labels):
    """Check annotation label.

    Parameters
    ----------
    labels : str {"indicator", "score", "int_label"}
        annotation labels

    Returns
    -------
    label : str
        Checked annotation label.
    """
    valid_labels = ["indicator", "score", "int_label"]
    if labels not in valid_labels:
        raise ValueError(f"`labels` must be in: {valid_labels}, but found: {labels}.")
