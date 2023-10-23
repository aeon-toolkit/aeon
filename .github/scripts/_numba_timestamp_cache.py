# save_timestamps.py
import os
import sys


def _apply_timestamps(directory, reference_file):
    # get the python file stamps
    reference_mtime = os.path.getmtime(reference_file)

    for root, _, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            os.utime(filepath, (reference_mtime, reference_mtime))


if __name__ == "__main__":
    """Script to save and compare the timestamps of the numba cache.

    Parameters
    ----------
    [0]: Script path
        Path to this script.
    [1]: Numba cache directory
        Directory of numba cache.
    [2]: Reference file
        File to set timestamp to match
    """
    print("Running script")  # noqa: T001, T201
    directory = sys.argv[1]
    reference_file = sys.argv[2]
    _apply_timestamps(directory, reference_file)
