import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from data_utils import load_dataset, load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MultiROCKET model on radar trajectory .xls files."
    )
    parser.add_argument(
        "--data-dir",
        default="mydataset/radar_augv3",
        help="Root dataset directory with class subfolders.",
    )
    parser.add_argument(
        "--model-path",
        default="radar_multirocket/output/model.joblib",
        help="Path to trained model joblib.",
    )
    parser.add_argument(
        "--split-file",
        default="radar_multirocket/output/split.json",
        help="Split file created by train.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    split_file = Path(args.split_file)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    split_payload = load_json(split_file)
    test_items = [
        {"path": str(data_dir / item["path"]), "label": item["label"]}
        for item in split_payload.get("test", [])
    ]
    if not test_items:
        raise ValueError("Split file does not contain test items")

    meta = {}
    meta_path = split_file.parent / "meta.json"
    if meta_path.is_file():
        meta = load_json(meta_path)

    X_test, y_test, _, _, _ = load_dataset(
        data_dir,
        class_names=meta.get("class_names"),
        extensions=tuple(meta.get("extensions", [".xls"]))
        if meta
        else (".xls",),
        pad_mode=meta.get("pad_mode", "pad"),
        pad_value=meta.get("pad_value", 0.0),
        max_length=meta.get("target_length"),
        nan_strategy=meta.get("nan_strategy", "zero"),
        file_list=test_items,
    )

    clf = joblib.load(model_path)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(report)
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
