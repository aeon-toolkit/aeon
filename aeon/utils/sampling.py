"""Functions to sample aeon datasets.

Used in experiments to get deterministic resamples.
"""

import random


def random_partition(n, k=2, seed=42):
    """Construct a uniformly random partition, iloc reference.

    Parameters
    ----------
    n : int
        size of set to partition
    k : int, optional, default=2
        number of sets to partition into
    seed : int
        random seed, used in random.shuffle

    Returns
    -------
    parts : list of list of int
        elements of `parts` are lists of iloc int indices between 0 and n-1
        elements of `parts` are of length floor(n / k) or ceil(n / k)
        elements of `parts`, as sets, are disjoint partition of [0, ..., n-1]
        elements of elements of `parts` are in no particular order
        `parts` is sampled uniformly at random, subject to the above properties
    """
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)

    parts = []
    for i in range(k):
        d = round(len(idx) / (k - i))
        parts += [idx[:d]]
        idx = idx[d:]

    return parts
