import os


class safe_import:

    def __enter__(self):
        self.error = False
        return self

    def __exit__(self, type_, value, traceback):
        if type_ is not None:
            self.error = True
            suppress = not (
                os.getenv("SCIPY_ALLOW_BENCH_IMPORT_ERRORS", "1").lower()
                in ("0", "false")
                or not issubclass(type_, ImportError)
            )
            return suppress


class Benchmark:
    """
    Base class with sensible options
    """
