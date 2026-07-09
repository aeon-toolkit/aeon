[//]: # (This was adapted from: https://github.com/scipy/scipy/tree/main/benchmarks)
# aeon Time Series Benchmarks

Benchmarking aeon with Airspeed Velocity.

## Usage

Airspeed Velocity manages building and Python environments by itself, unless told
otherwise. Some of the benchmarking features in `spin` also tell ASV to use the aeon
compiled by `spin`. To run the benchmarks, you will need to install the "dev"
dependencies of aeon:

```bash
pip install --editble .[dev]
# NOTE: If the above fails, try running pip install --editable ".[dev]"
```

Run a benchmark against currently checked-out aeon version (don't record the result):

```bash
spin bench --submodule classification.distance_based
```

Compare change in benchmark results with another branch:

```bash
spin bench --compare main --submodule classification.distance_based
```

Run ASV commands directly (note, this will not set env vars for `ccache` and disabling BLAS/LAPACK multi-threading, as `spin` does):

```bash
cd benchmarks
asv run --skip-existing-commits --steps 10 ALL
asv publish
asv preview
```

More on how to use `asv` can be found in [ASV documentation](https://asv.readthedocs.io/). Command-line help is available as usual via `asv --help` and `asv run --help`.

## Writing benchmarks

See [ASV documentation](https://asv.readthedocs.io/) for the basics on how to write benchmarks.

Some things to consider:

- When importing things from aeon on the top of the test files, do it as:

  ```python
  from .common import safe_import

  with safe_import():
      from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
  ```

  The benchmark files need to be importable also when benchmarking old versions of aeon. The benchmarks themselves don't need any guarding against missing features — only the top-level imports.

- Try to keep the runtime of the benchmark reasonable.

- Use ASV's `time_` methods for benchmarking times rather than cooking up time measurements via `time.clock`, even if it requires some juggling when writing the benchmark.

- Preparing arrays etc., should generally be put in the `setup` method rather than the `time_` methods, to avoid counting preparation time together with the time of the benchmarked operation.

- Use `run_monitored` from `common.py` if you need to measure memory usage.

- Benchmark versioning: by default `asv` invalidates old results when there is any code change in the benchmark routine or in setup/setup_cache.

  This can be controlled manually by setting a fixed benchmark version number, using the `version` attribute. See [ASV documentation](https://asv.readthedocs.io/) for details.

  If set manually, the value needs to be changed manually when old results should be invalidated. In case you want to preserve previous benchmark results when the benchmark did not previously have a manual `version` attribute, the automatically computed default values can be found in `results/benchmark.json`.

- Benchmark attributes such as `params` and `param_names` must be the same regardless of whether some features are available, or e.g. AEON_XSLOW=1 is set.

  Instead, benchmarks that should not be run can be skipped by raising `NotImplementedError` in `setup()`.
