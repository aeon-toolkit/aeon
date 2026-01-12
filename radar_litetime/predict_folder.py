import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from data_utils import collect_files_recursive, load_files, load_json

from aeon.classification.deep_learning import LITETimeClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict LITETime classes for .xls files in a folder (recursive)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Root directory to scan for .xls files.",
    )
    parser.add_argument(
        "--model-dir",
        default="radar_litetime/output",
        help="Directory containing saved .keras models.",
    )
    parser.add_argument(
        "--model-files",
        default="",
        help="Model file names, comma-separated (overrides meta.json).",
    )
    parser.add_argument(
        "--meta-file",
        default="radar_litetime/output/meta.json",
        help="Meta file created by train.py.",
    )
    parser.add_argument(
        "--classes",
        default="",
        help="Class names, comma-separated (required if meta.json missing).",
    )
    parser.add_argument(
        "--extensions",
        default="",
        help="File extensions to scan, comma-separated (default: from meta or .xls).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Target length for each message; overrides meta.json target_length.",
    )
    parser.add_argument(
        "--pad-mode",
        choices=["pad", "truncate", "error"],
        default=None,
        help="Length mismatch handling when target length is fixed.",
    )
    parser.add_argument("--pad-value", type=float, default=None, help="Padding value.")
    parser.add_argument(
        "--nan-strategy",
        choices=["zero", "ffill", "drop"],
        default=None,
        help="NaN handling strategy.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output.",
    )
    return parser.parse_args()


def _resolve_model_paths(model_dir, model_files):
    paths = []
    for item in model_files:
        candidate = Path(item)
        if candidate.is_file():
            paths.append(candidate)
        else:
            paths.append(Path(model_dir) / item)
    return paths


def _parse_csv_arg(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    meta = {}
    meta_file = Path(args.meta_file)
    if meta_file.is_file():
        meta = load_json(meta_file)

    if args.model_files:
        model_files = _parse_csv_arg(args.model_files)
    else:
        model_files = meta.get("model_files", [])
    if not model_files:
        raise ValueError("No model files provided; use --model-files or meta.json")

    model_paths = _resolve_model_paths(args.model_dir, model_files)
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

    if args.classes:
        class_names = _parse_csv_arg(args.classes)
    else:
        class_names = meta.get("class_names", [])
    if not class_names:
        raise ValueError("Class names missing; use --classes or meta.json")

    if args.extensions:
        extensions = tuple(_parse_csv_arg(args.extensions))
    else:
        extensions = tuple(meta.get("extensions", [])) or (".xls",)

    pad_mode = args.pad_mode or meta.get("pad_mode", "pad")
    pad_value = (
        args.pad_value if args.pad_value is not None else meta.get("pad_value", 0.0)
    )
    nan_strategy = args.nan_strategy or meta.get("nan_strategy", "zero")

    target_length = (
        args.length if args.length is not None else meta.get("target_length")
    )
    if target_length is None:
        raise ValueError(
            "Target length is required; use --length or meta.json target_length"
        )

    file_paths = collect_files_recursive(input_dir, extensions=extensions)
    if not file_paths:
        raise ValueError(f"No input files found with extensions: {extensions}")

    return_lengths = pad_mode == "error"
    if return_lengths:
        X, paths, used_length, lengths = load_files(
            file_paths,
            pad_mode=pad_mode,
            pad_value=pad_value,
            target_length=target_length,
            nan_strategy=nan_strategy,
            show_progress=not args.no_progress,
            return_lengths=True,
        )
        mismatched = [p for p, l in zip(paths, lengths) if l != target_length]
        if mismatched:
            sample = "\n".join(mismatched[:5])
            raise ValueError(
                "Found files with lengths not equal to target_length. "
                f"Example files:\n{sample}"
            )
    else:
        X, paths, used_length = load_files(
            file_paths,
            pad_mode=pad_mode,
            pad_value=pad_value,
            target_length=target_length,
            nan_strategy=nan_strategy,
            show_progress=not args.no_progress,
            return_lengths=False,
        )

    clf = LITETimeClassifier.load_model(
        model_path=[str(p) for p in model_paths],
        classes=np.array(class_names),
    )
    y_pred = clf.predict(X)

    for path, label in zip(paths, y_pred):
        print(f"{path}\t{label}")

    counts = Counter(y_pred)
    print("Class counts:")
    for name in class_names:
        print(f"{name}\t{counts.get(name, 0)}")
    for name in sorted(counts.keys()):
        if name not in class_names:
            print(f"{name}\t{counts[name]}")

    print(f"Target length used: {used_length}")
    print(f"Pad mode: {pad_mode}, pad value: {pad_value}, nan strategy: {nan_strategy}")


if __name__ == "__main__":
    main()
