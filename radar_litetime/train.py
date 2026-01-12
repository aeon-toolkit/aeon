import argparse
import glob
import time
from pathlib import Path

import numpy as np
from data_utils import load_dataset, save_json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from aeon.classification.deep_learning import LITETimeClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用雷达轨迹 .xls/.xlsx 文件训练 LITETime 分类器。"
    )
    parser.add_argument(
        "--data-dir",
        default="mydataset/radar_augv3",
        help="数据集根目录，内部包含各类别子目录。",
    )
    parser.add_argument(
        "--output-dir",
        default="radar_litetime/output",
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
        "--val-size",
        type=float,
        default=0.1,
        help="验证集比例（从训练集中划分，0 表示不使用验证集）。",
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
        "--n-classifiers", type=int, default=5, help="LITETime 子模型数量。"
    )
    parser.add_argument(
        "--use-litemv",
        action="store_true",
        help="启用 LITEMV（多变量）结构。",
    )
    parser.add_argument("--n-filters", type=int, default=32, help="卷积层滤波器数量。")
    parser.add_argument("--kernel-size", type=int, default=40, help="卷积核大小。")
    parser.add_argument("--strides", type=int, default=1, help="卷积步幅。")
    parser.add_argument("--activation", default="relu", help="激活函数。")

    parser.add_argument("--batch-size", type=int, default=64, help="训练 batch size。")
    parser.add_argument(
        "--use-mini-batch-size",
        action="store_true",
        help="自动设置 mini batch size（适合大数据）。",
    )
    parser.add_argument("--n-epochs", type=int, default=100, help="训练轮数。")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示训练过程日志。",
    )

    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="保存训练过程中损失最小的模型。",
    )
    parser.add_argument(
        "--save-last-model",
        action="store_true",
        help="保存最后一轮训练的模型。",
    )
    parser.add_argument(
        "--best-file-name",
        default="litetime_best",
        help="最佳模型保存文件名前缀。",
    )
    parser.add_argument(
        "--last-file-name",
        default="litetime_last",
        help="最后模型保存文件名前缀。",
    )

    return parser.parse_args()


def _split_train_val(
    X_train_full, y_train_full, paths_train_full, val_size, random_state
):
    if val_size and val_size > 0:
        return train_test_split(
            X_train_full,
            y_train_full,
            paths_train_full,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train_full,
        )
    return (
        X_train_full,
        np.empty((0,)),
        y_train_full,
        np.empty((0,)),
        paths_train_full,
        [],
    )


def _collect_model_files(output_dir, best_prefix, last_prefix, save_best, save_last):
    if save_best:
        return sorted(output_dir.glob(f"{best_prefix}*.keras"))
    if save_last:
        return sorted(output_dir.glob(f"{last_prefix}*.keras"))
    return []


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

    (
        X_train,
        X_val,
        y_train,
        y_val,
        paths_train,
        paths_val,
    ) = _split_train_val(
        X_train_full, y_train_full, paths_train_full, args.val_size, args.random_state
    )

    file_path = str(output_dir) + "/"
    clf = LITETimeClassifier(
        n_classifiers=args.n_classifiers,
        use_litemv=args.use_litemv,
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        strides=args.strides,
        activation=args.activation,
        batch_size=args.batch_size,
        use_mini_batch_size=args.use_mini_batch_size,
        n_epochs=args.n_epochs,
        random_state=args.random_state,
        verbose=args.verbose,
        save_best_model=args.save_best_model,
        save_last_model=args.save_last_model,
        file_path=file_path,
        best_file_name=args.best_file_name,
        last_file_name=args.last_file_name,
    )

    print("Fitting LITETimeClassifier...")
    fit_start = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start
    print(f"Training finished in {fit_seconds:.2f}s")

    print("Predicting on test set...")
    pred_start = time.perf_counter()
    y_pred = clf.predict(X_test)
    pred_seconds = time.perf_counter() - pred_start
    print(f"Prediction finished in {pred_seconds:.2f}s")

    report_text = classification_report(y_test, y_pred, digits=4)
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    print(report_text)
    print("Confusion matrix:")
    print(matrix)

    val_report = None
    val_matrix = None
    if len(y_val) > 0:
        print("Predicting on validation set...")
        y_val_pred = clf.predict(X_val)
        val_report = classification_report(
            y_val, y_val_pred, digits=4, output_dict=True
        )
        val_matrix = confusion_matrix(y_val, y_val_pred).tolist()

    model_files = _collect_model_files(
        output_dir,
        args.best_file_name,
        args.last_file_name,
        args.save_best_model,
        args.save_last_model,
    )
    model_files = [p.name for p in model_files]

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
            "n_classifiers": args.n_classifiers,
            "use_litemv": args.use_litemv,
            "n_filters": args.n_filters,
            "kernel_size": args.kernel_size,
            "strides": args.strides,
            "activation": args.activation,
            "batch_size": args.batch_size,
            "use_mini_batch_size": args.use_mini_batch_size,
            "n_epochs": args.n_epochs,
            "random_state": args.random_state,
            "save_best_model": args.save_best_model,
            "save_last_model": args.save_last_model,
            "best_file_name": args.best_file_name,
            "last_file_name": args.last_file_name,
        },
        "model_files": model_files,
        "metrics": {
            "classification_report": report_dict,
            "confusion_matrix": matrix,
            "val_classification_report": val_report,
            "val_confusion_matrix": val_matrix,
        },
    }
    save_json(output_dir / "meta.json", meta)

    if model_files:
        print("Saved model files:")
        for name in model_files:
            print(f"- {output_dir / name}")
    else:
        print("No model files saved. Enable --save-best-model or --save-last-model.")


if __name__ == "__main__":
    main()
