"""Custom logic for numba caching during the cicd run."""

import os
import pickle
import subprocess

import numba.core.caching

if os.environ.get("CICD_RUNNING") == "1":  # pragma: no cover

    def get_invalid_numba_files():
        """Get the files that have been changed since the last commit.

        This function is used to get the files that have been changed. This is needed
        because custom logic to save the numba cache has been implemented for numba.
        This function returns the file names that have been changed and if they appear
        in here any numba functions cache are invalidated.

        Returns
        -------
        list
            List of file names that have been changed.
        """
        subprocess.run(["git", "fetch", "origin", "main"], check=True)

        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main"],
            check=True,
            capture_output=True,
            text=True,  # Makes the output text instead of bytes
        )

        files = result.stdout.split("\n")

        files = [file for file in files if file]

        clean_files = []

        for temp in files:
            if temp.endswith(".py"):
                clean_files.append((temp.split("/")[-1]).strip(".py"))

        return clean_files

    # Retry the git fetch and git diff commands in case of failure
    retry = 0
    while retry < 3:
        try:
            CHANGED_FILES = get_invalid_numba_files()
            break
        except subprocess.CalledProcessError:
            retry += 1

    # If the retry count is reached, raise an error
    if retry == 3:
        raise Exception("Failed to get the changed files from git.")

    def custom_load_index(self):
        """Overwrite load index method for numba.

        This is used to overwrite the numba internal logic to allow for caching during
        the cicd run. Numba traditionally checks the timestamp of the file and if it
        has changed it invalidates the cache. This is not ideal for the cicd run as
        the cache restore is always before the files (since it is cloned in) and
        thus the cache is always invalidated. This custom method ignores the timestamp
        element and instead just checks the file name. This isn't as fine grained as
        numba but it is better to invalidate more and make sure the right function has
        been compiled than try to be too clever and miss some.

        Returns
        -------
        dict
            Dictionary of the cached functions.
        """
        try:
            with open(self._index_path, "rb") as f:
                version = pickle.load(f)
                data = f.read()
        except FileNotFoundError:
            return {}
        if version != self._version:
            return {}
        stamp, overloads = pickle.loads(data)
        cache_filename = self._index_path.split("/")[-1].split("-")[0].split(".")[0]
        if stamp[1] != self._source_stamp[1] or cache_filename in CHANGED_FILES:
            return {}
        else:
            return overloads

    original_load_index = numba.core.caching.IndexDataCacheFile._load_index
    numba.core.caching.IndexDataCacheFile._load_index = custom_load_index

    # Force all numba functions to be cached
    original_jit = numba.core.decorators._jit

    def custom_njit(*args, **kwargs):
        """Force jit to cache.

        This is used for libraries like stumpy that doesn't cache by default. This
        function will force all functions running to be cache'd
        """
        target = kwargs["targetoptions"]
        # This target can't be cached
        if "no_cpython_wrapper" not in target:
            kwargs["cache"] = True
        return original_jit(*args, **kwargs)

    # Overwrite the jit function with the custom version
    numba.core.decorators._jit = custom_njit
