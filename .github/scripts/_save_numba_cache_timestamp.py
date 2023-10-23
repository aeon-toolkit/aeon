import os
import pickle


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
        File to save the dict of time stamps to.
    """
    timestamps = {}
    for root, _, files in os.walk(directory):
        for name in files:
            filepath = os.path.join(root, name)
            with open(filepath, "rb") as f:
                data = f.read()
            stamp, _ = pickle.loads(data)
            timestamps[filepath] = stamp

    with open(output_file, "wb") as f:
        pickle.dump(timestamps, f)
