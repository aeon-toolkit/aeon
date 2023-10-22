# save_timestamps.py
import os
import pickle
import sys


def _save_timestamps(directory, output_file):
    """Script that records the chache'd timestamps of all files in a directory.

    When the numba files are cache'd it does not store the last time the file was
    updated. As such when the cache is loaded it will be invalidated because it
    is missing the timestamp. This script records the timestamps of all files in
    the numba cache so they can be correctly reloaded after loading the cahce.

    Parameters
    ----------
    directory : str
        Directory to record timestamps for.
    output_file : str
        File to save the timestamps to.
    """
    timestamps = {}
    for root, _, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            timestamps[filepath] = os.path.getmtime(filepath)

    with open(output_file, "wb") as f:
        pickle.dump(timestamps, f)


def _compare_timestamps(directory, input_file):
    with open(input_file, "rb") as f:
        old_timestamps = pickle.load(f)

    mismatches = []
    for root, _, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            if filepath in old_timestamps:
                current_mtime = os.path.getmtime(filepath)
                if current_mtime != old_timestamps[filepath]:
                    mismatches.append(
                        (filepath, old_timestamps[filepath], current_mtime)
                    )

    if mismatches:
        print(f"Found {len(mismatches)} mismatches!")  # noqa: T001, T201
        for mismatch in mismatches:
            print_val = (
                f"File: {mismatch[0]} | Cached mtime: {mismatch[1]} | "
                f"Current mtime: {mismatch[2]}"
            )
            print(print_val)  # noqa: T001, T201
    else:
        print("All timestamps match!")  # noqa: T001, T201


if __name__ == "__main__":
    """Script to save and compare the timestamps of the numba cache.

    Parameters
    ----------
    [0]: Script path
        Path to this script.
    [1]: Operation
        Either "save" or "compare" depending on what you want to do.
    [2]: Numba cache directory
        Directory to save/compare the timestamps for.
    [3]: Output file
        File to save the timestamps to.
    """
    print("Running script")  # noqa: T001, T201
    operation = sys.argv[1]
    directory = sys.argv[2]
    output_file = sys.argv[3]
    if sys.argv[1] == "save":
        _save_timestamps(directory, output_file)
    else:
        _compare_timestamps(directory, output_file)
