import os
import pickle
import sys


def _update_pickle_timestamp(pickle_file_path, new_stamp):
    # Check if the pickle file exists
    if not os.path.exists(pickle_file_path):
        print(f"Pickle file not found: {pickle_file_path}")  # noqa: T001, T201
        return

    # Step 1: Deserialize the data from the pickle file
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)

    # Check if the data is in the expected format (i.e., a tuple)
    if not isinstance(data, tuple) or len(data) != 2:
        return

    # Extract the current stamp and overloads
    _, overloads = data

    # Step 2: Create a new tuple with the updated stamp
    updated_data = (new_stamp, overloads)

    # Step 3: Reserialize the data and save it back to the pickle file
    with open(pickle_file_path, "wb") as file:
        pickle.dump(updated_data, file)


def _apply_timestamps(directory, reference_file):
    # Get the modification time of the reference file
    reference_mtime = os.path.getmtime(reference_file)
    print(f"Setting pickle timestamps to {reference_mtime}")  # noqa: T001, T201

    # Walk through the directory and update pickle files
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".pkl"):  # Check for pickle files
                pickle_file_path = os.path.join(root, name)
                print(f"Updating pickle file: {pickle_file_path}")  # noqa: T001, T201
                _update_pickle_timestamp(pickle_file_path, reference_mtime)


# Usage
# def _apply_timestamps(directory, reference_file):
#     # get the python file stamps
#     reference_mtime = os.path.getmtime(reference_file)
#     print(f"Setting timestamps to {reference_mtime}")  # noqa: T001, T201
#
#     for root, _, files in os.walk(directory):
#         for name in files:
#             filepath = os.path.join(root, name)
#             print(f"Setting timestamp for {filepath}")  # noqa: T001, T201
#             os.utime(filepath, (reference_mtime, reference_mtime))


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
