import argparse
from pathlib import Path

import numpy as np
from data_utils import load_dataset, load_json
from sklearn.metrics import classification_report, confusion_matrix

from aeon.classification.deep_learning import LITETimeClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LITETime model on radar trajectory .xls/.xlsx files."
    )
    parser.add_argument(
        "--data-dir",
        default="mydataset/radar_augv3",
        help="Root dataset directory with class subfolders.",
    )
    parser.add_argument(
        "--model-dir",
        default="radar_litetime/output",
        help="Directory containing saved .keras models.",
    )
    parser.add_argument(
        "--model-files",
        default="",
        help="模型文件名列表，逗号分隔（优先级高于 meta.json）。",
    )
    parser.add_argument(
        "--split-file",
        default="radar_litetime/output/split.json",
        help="Split file created by train.py.",
    )
    parser.add_argument(
        "--meta-file",
        default="radar_litetime/output/meta.json",
        help="Meta file created by train.py.",
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


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    split_file = Path(args.split_file)
    meta_file = Path(args.meta_file)

    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    meta = {}
    if meta_file.is_file():
        meta = load_json(meta_file)

    split_payload = load_json(split_file)
    test_items = [
        {"path": str(data_dir / item["path"]), "label": item["label"]}
        for item in split_payload.get("test", [])
    ]
    if not test_items:
        raise ValueError("Split file does not contain test items")

    if args.model_files:
        model_files = [
            name.strip() for name in args.model_files.split(",") if name.strip()
        ]
    else:
        model_files = meta.get("model_files", [])

    if not model_files:
        raise ValueError("No model files provided; use --model-files or meta.json")

    model_paths = _resolve_model_paths(args.model_dir, model_files)
    for path in model_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

    class_names = meta.get("class_names")
    if not class_names:
        raise ValueError(
            "meta.json missing class_names; please re-train or provide meta"
        )

    X_test, y_test, _, _, _ = load_dataset(
        data_dir,
        class_names=class_names,
        extensions=tuple(meta.get("extensions", [".xls"])) if meta else (".xls",),
        pad_mode=meta.get("pad_mode", "pad"),
        pad_value=meta.get("pad_value", 0.0),
        max_length=meta.get("target_length"),
        nan_strategy=meta.get("nan_strategy", "zero"),
        file_list=test_items,
    )

    clf = LITETimeClassifier.load_model(
        model_path=[str(p) for p in model_paths],
        classes=np.array(class_names),
    )
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(report)
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
