"""Benchmark distance functions (pairwise focused)."""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

DEFAULT_DISTANCE_MODULES = ("aeon.distances.pointwise._{name}",)

DISTANCE_PARAMS: dict[str, list[float]] = {
    "minkowski": [2.0],
}


def generate_collection(instances: int, channels: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    if channels == 1:
        return rng.random((instances, length))
    return rng.random((instances, channels, length))


def benchmark_pairwise_distance(
    dist_name: str,
    pairwise_func,
    X: np.ndarray,
    iterations: int,
    params: list[float],
) -> dict[str, float]:
    if dist_name == "minkowski":
        try:
            pairwise_func(X, p=2.0)
        except TypeError:
            pairwise_func(X, None, 1, 2.0)
    elif params:
        pairwise_func(X, None, 1, *params)
    else:
        pairwise_func(X)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        if dist_name == "minkowski":
            try:
                pairwise_func(X, p=2.0)
            except TypeError:
                pairwise_func(X, None, 1, 2.0)
        elif params:
            pairwise_func(X, None, 1, *params)
        else:
            pairwise_func(X)
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    timings_arr = np.array(timings)
    return {
        "mean_ms": float(np.mean(timings_arr)),
        "std_ms": float(np.std(timings_arr)),
        "min_ms": float(np.min(timings_arr)),
        "max_ms": float(np.max(timings_arr)),
    }


def run_benchmarks(
    distances: list[str],
    lengths: list[int],
    instances: list[int],
    channels_list: list[int],
    iterations: int,
) -> list[dict[str, Any]]:
    results = []
    distance_modules: dict[str, tuple[Any, Any]] = {}

    for dist_name in distances:
        for module_template in DEFAULT_DISTANCE_MODULES:
            module_name = module_template.format(name=dist_name)
            try:
                module = __import__(module_name, fromlist=[dist_name])
                pairwise_func = getattr(module, f"{dist_name}_pairwise_distance")
                distance_modules[dist_name] = (None, pairwise_func)
                break
            except (ImportError, AttributeError):
                continue
        if dist_name not in distance_modules:
            raise ImportError(f"Could not import {dist_name}")

    print("\nBenchmarking pairwise distances...")
    for dist_name, (_, pairwise_func) in distance_modules.items():
        params = DISTANCE_PARAMS.get(dist_name, [])
        for n_instances in instances:
            for length in lengths:
                for channels in channels_list:
                    X = generate_collection(n_instances, channels, length)
                    stats = benchmark_pairwise_distance(
                        dist_name, pairwise_func, X, iterations, params
                    )

                    results.append(
                        {
                            "test_type": "pairwise_distance",
                            "distance": dist_name,
                            "channels": channels,
                            "length": length,
                            "instances": n_instances,
                            **stats,
                        }
                    )

                    print(
                        f"  {dist_name} (instances={n_instances}, length={length}, "
                        f"channels={channels}): {stats['mean_ms']:.4f} ± {stats['std_ms']:.4f} ms"
                    )

    return results


def save_results_csv(results: list[dict[str, Any]], output_path: str) -> None:
    import csv

    if not results:
        return

    fieldnames = [
        "test_type",
        "distance",
        "channels",
        "length",
        "instances",
        "mean_ms",
        "std_ms",
        "min_ms",
        "max_ms",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark aeon distance functions")
    parser.add_argument(
        "--distances",
        nargs="+",
        default=["squared", "euclidean", "manhattan", "minkowski"],
        help="Distance functions to benchmark",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[100, 200],
        help="Time series lengths to test",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        type=int,
        default=[100, 250, 500],
        help="Number of instances for pairwise tests",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[1, 5],
        help="Number of channels (1=univariate)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Distance Function Benchmark")
    print("=" * 70)
    print(f"Distances: {', '.join(args.distances)}")
    print(f"Lengths: {', '.join(map(str, args.lengths))}")
    print(f"Instances: {', '.join(map(str, args.instances))}")
    print(f"Channels: {', '.join(map(str, args.channels))}")
    print(f"Iterations: {args.iterations}")
    print("=" * 70)

    results = run_benchmarks(
        distances=args.distances,
        lengths=args.lengths,
        instances=args.instances,
        channels_list=args.channels,
        iterations=args.iterations,
    )

    save_results_csv(results, args.output_csv)
    print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
