import argparse
import time
from itertools import product
from pathlib import Path

import joblib
from data_utils import load_dataset, save_json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from aeon.classification.convolution_based import MultiRocketClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用雷达轨迹 .xls 文件训练 MultiROCKET 分类器。"
    )
    parser.add_argument(
        "--data-dir",
        default="mydataset/radar_augv3",
        help="数据集根目录，内部包含各类别子目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="radar_multirocket/output",
        help="模型与元数据输出目录。",
    )
    parser.add_argument(
        "--classes",
        default="bird,uav",
        help="类别文件夹名称，逗号分隔。",
    )
    parser.add_argument(
        "--extensions",
        default=".xls",
        help="要读取的文件扩展名，逗号分隔。",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例。")
    parser.add_argument(
        "--val-size", type=float, default=0.1, help="验证集比例（从训练集中划分）。"
    )
    parser.add_argument("--random-state", type=int, default=42, help="随机种子。")

    parser.add_argument(
        "--pad-mode",
        choices=["pad", "truncate", "error"],
        default="pad",
        help="长度不一致时的处理方式：pad 补齐、truncate 截断、error 报错。",
    )
    parser.add_argument(
        "--pad-value", type=float, default=0.0, help="补齐时使用的数值。"
    )
    parser.add_argument(
        "--max-length", type=int, default=None, help="强制截断/补齐到固定长度。"
    )
    parser.add_argument(
        "--nan-strategy",
        choices=["zero", "ffill", "drop"],
        default="zero",
        help="NaN 处理策略：zero 填 0、ffill 前向填充、drop 删除含 NaN 行。",
    )

    parser.add_argument(
        "--n-kernels", type=int, default=10000, help="MultiROCKET 核数量。"
    )
    parser.add_argument(
        "--max-dilations-per-kernel",
        type=int,
        default=32,
        help="每个核的最大膨胀数。",
    )
    parser.add_argument(
        "--n-features-per-kernel",
        type=int,
        default=4,
        help="每个核提取的特征数量。",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="并行任务数，-1 表示全部核心。"
    )
    parser.add_argument(
        "--enable-search",
        action="store_true",
        help="启用简单网格搜索（会显著增加训练时间）。",
    )
    parser.add_argument(
        "--search-n-kernels",
        default="10000,20000,50000",
        help="网格搜索 n_kernels 值，逗号分隔。",
    )
    parser.add_argument(
        "--search-max-dilations",
        default="16,32,64",
        help="网格搜索 max_dilations_per_kernel 值，逗号分隔。",
    )
    parser.add_argument(
        "--search-n-features",
        default="4,8",
        help="网格搜索 n_features_per_kernel 值，逗号分隔。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    extensions = tuple(ext.strip() for ext in args.extensions.split(",") if ext.strip())

    X, y, paths, class_names, target_length = load_dataset(
        data_dir,
        class_names=class_names,
        extensions=extensions,
        pad_mode=args.pad_mode,
        pad_value=args.pad_value,
        max_length=args.max_length,
        nan_strategy=args.nan_strategy,
    )

    X_train_full, X_test, y_train_full, y_test, paths_train_full, paths_test = (
        train_test_split(
            X,
            y,
            paths,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    )

    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X_train_full,
        y_train_full,
        paths_train_full,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_train_full,
    )

    def train_and_eval(n_kernels, max_dilations, n_features):
        clf = MultiRocketClassifier(
            n_kernels=n_kernels,
            max_dilations_per_kernel=max_dilations,
            n_features_per_kernel=n_features,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        fit_start = time.perf_counter()  # Start time for training.
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start  # Training duration in seconds.
        y_val_pred = clf.predict(X_val)
        report = classification_report(y_val, y_val_pred, digits=4, output_dict=True)
        val_acc = report["accuracy"]
        return clf, val_acc, fit_seconds

    best_clf = None
    best_acc = -1.0
    best_params = {
        "n_kernels": args.n_kernels,
        "max_dilations_per_kernel": args.max_dilations_per_kernel,
        "n_features_per_kernel": args.n_features_per_kernel,
    }

    if args.enable_search:
        search_n_kernels = [
            int(v) for v in args.search_n_kernels.split(",") if v.strip()
        ]
        search_max_dilations = [
            int(v) for v in args.search_max_dilations.split(",") if v.strip()
        ]
        search_n_features = [
            int(v) for v in args.search_n_features.split(",") if v.strip()
        ]
        combos = list(
            product(search_n_kernels, search_max_dilations, search_n_features)
        )
        total = len(combos)
        print(f"Start grid search: {total} configs")
        for i, (n_kernels, max_dilations, n_features) in enumerate(combos, start=1):
            print(
                f"[{i}/{total}] training n_kernels={n_kernels}, "
                f"max_dilations={max_dilations}, n_features={n_features}"
            )
            clf, val_acc, fit_seconds = train_and_eval(
                n_kernels, max_dilations, n_features
            )
            print(
                f"[{i}/{total}] val accuracy={val_acc:.4f}, train_time={fit_seconds:.2f}s"
            )
            if val_acc > best_acc:
                best_acc = val_acc
                best_clf = clf
                best_params = {
                    "n_kernels": n_kernels,
                    "max_dilations_per_kernel": max_dilations,
                    "n_features_per_kernel": n_features,
                }
        print(f"Best val accuracy={best_acc:.4f} with {best_params}")
        clf = best_clf
    else:
        clf = MultiRocketClassifier(
            n_kernels=args.n_kernels,
            max_dilations_per_kernel=args.max_dilations_per_kernel,
            n_features_per_kernel=args.n_features_per_kernel,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        print("Fitting MultiRocketClassifier...")
        fit_start = time.perf_counter()  # Start time for training.
        clf.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start  # Training duration in seconds.
        print(f"Training finished in {fit_seconds:.2f}s")

    print("Predicting on test set...")
    pred_start = time.perf_counter()  # Start time for prediction.
    y_pred = clf.predict(X_test)
    pred_seconds = time.perf_counter() - pred_start  # Prediction duration in seconds.
    print(f"Prediction finished in {pred_seconds:.2f}s")
    report = classification_report(y_test, y_pred, digits=4)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(report)
    print("Confusion matrix:")
    print(matrix)

    model_path = output_dir / "model.joblib"
    joblib.dump(clf, model_path)

    split_payload = {
        "train": [
            {"path": str(Path(p).relative_to(data_dir)), "label": label}
            for p, label in zip(paths_train, y_train)
        ],
        "val": [
            {"path": str(Path(p).relative_to(data_dir)), "label": label}
            for p, label in zip(paths_val, y_val)
        ],
        "test": [
            {"path": str(Path(p).relative_to(data_dir)), "label": label}
            for p, label in zip(paths_test, y_test)
        ],
    }
    save_json(output_dir / "split.json", split_payload)

    meta = {
        "data_dir": str(data_dir),
        "class_names": class_names,
        "extensions": list(extensions),
        "pad_mode": args.pad_mode,
        "pad_value": args.pad_value,
        "max_length": args.max_length,
        "target_length": target_length,
        "nan_strategy": args.nan_strategy,
        "model": {
            "n_kernels": args.n_kernels,
            "max_dilations_per_kernel": args.max_dilations_per_kernel,
            "n_features_per_kernel": args.n_features_per_kernel,
            "n_jobs": args.n_jobs,
            "random_state": args.random_state,
        },
        "search": {
            "enabled": args.enable_search,
            "val_size": args.val_size,
            "best_params": best_params,
            "best_val_accuracy": best_acc if args.enable_search else None,
        },
        "metrics": {
            "classification_report": report,
            "confusion_matrix": matrix,
        },
    }
    save_json(output_dir / "meta.json", meta)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
