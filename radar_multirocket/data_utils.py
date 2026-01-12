import json
from pathlib import Path

import numpy as np


def _read_excel(file_path, nan_strategy):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to read .xls files. Install with: pip install pandas xlrd"
        ) from exc

    file_path = Path(file_path)
    engine = "xlrd" if file_path.suffix.lower() == ".xls" else None
    try:
        # pandas read_excel/read_csv are used for file parsing
        df = pd.read_excel(file_path, engine=engine)
    except Exception:
        if file_path.suffix.lower() != ".xls":
            raise
        df = pd.read_csv(file_path, encoding="gbk", sep=None, engine="python")
        engine = None
    df = df.select_dtypes(include=["number"]).copy()
    if df.empty:
        if engine is not None:
            df = pd.read_excel(file_path, engine=engine)
        else:
            df = pd.read_csv(file_path, encoding="gbk", sep=None, engine="python")
        df = df.apply(pd.to_numeric, errors="coerce")

    if df.empty:
        raise ValueError(f"No numeric columns found in {file_path}")

    if nan_strategy == "zero":
        df = df.fillna(0)
    elif nan_strategy == "ffill":
        df = df.fillna(method="ffill").fillna(method="bfill")
    elif nan_strategy == "drop":
        df = df.dropna(axis=0, how="any")
    else:
        raise ValueError(f"Unsupported nan_strategy: {nan_strategy}")

    values = df.to_numpy(dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def _collect_files(data_dir, class_names, extensions):
    data_dir = Path(data_dir)
    if class_names is None:
        class_names = [p.name for p in data_dir.iterdir() if p.is_dir()]
    class_names = sorted(class_names)

    items = []
    for label in class_names:
        class_dir = data_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        files = []
        for ext in extensions:
            files.extend(sorted(class_dir.glob(f"*{ext}")))
        if not files:
            raise FileNotFoundError(f"No data files for class: {label}")
        for file_path in files:
            items.append({"path": str(file_path), "label": label})
    return items, class_names


def _pad_or_truncate(series_list, pad_mode, pad_value, max_length=None):
    lengths = [arr.shape[0] for arr in series_list]
    if max_length is None:
        if pad_mode == "pad":
            target_length = max(lengths)
        elif pad_mode == "truncate":
            target_length = min(lengths)
        elif pad_mode == "error":
            target_length = lengths[0]
        else:
            raise ValueError(f"Unsupported pad_mode: {pad_mode}")
    else:
        target_length = int(max_length)

    if pad_mode == "error" and any(l != target_length for l in lengths):
        raise ValueError("Series lengths differ; use --pad-mode pad|truncate")

    processed = []
    for arr in series_list:
        length, channels = arr.shape
        if length == target_length:
            processed.append(arr)
            continue
        if length > target_length:
            processed.append(arr[:target_length])
            continue
        if length < target_length:
            pad_rows = target_length - length
            pad_block = np.full((pad_rows, channels), pad_value, dtype=arr.dtype)
            processed.append(np.vstack([arr, pad_block]))
            continue
    return processed, target_length


def load_dataset(
    data_dir,
    class_names=None,
    extensions=(".xls", ".xlsx"),
    pad_mode="pad",
    pad_value=0.0,
    max_length=None,
    nan_strategy="zero",
    file_list=None,
    show_progress=True,
):
    """Load dataset from a directory or file list.

    Args:
        data_dir (str or Path): Root directory with class subfolders.
        class_names (list[str] or None): Class names to load.
        extensions (tuple[str, ...]): Allowed file extensions.
        pad_mode (str): Padding/truncation mode.
        pad_value (float): Padding value.
        max_length (int or None): Override target length.
        nan_strategy (str): NaN handling strategy.
        file_list (list[dict] or None): Pre-collected file list.
        show_progress (bool): Whether to show a progress indicator.
    """
    if file_list is None:
        items, class_names = _collect_files(data_dir, class_names, extensions)
    else:
        items = file_list
        class_names = sorted({item["label"] for item in items})

    series_list = []
    labels = []
    paths = []

    items_iter = items  # Iterator for file loading.
    use_tqdm = False  # Whether tqdm progress bar is active.
    if show_progress:
        try:
            from tqdm import tqdm
        except Exception:
            pass
        else:
            items_iter = tqdm(items, desc="Loading files", unit="file")
            use_tqdm = True

    total = len(items)  # Total file count for plain progress output.
    for idx, item in enumerate(items_iter, start=1):
        if show_progress and not use_tqdm:
            print(f"\rLoading files {idx}/{total}", end="", flush=True)
        arr = _read_excel(item["path"], nan_strategy=nan_strategy)
        series_list.append(arr)
        labels.append(item["label"])
        paths.append(item["path"])
    if show_progress and not use_tqdm:
        print()

    series_list, target_length = _pad_or_truncate(
        series_list, pad_mode=pad_mode, pad_value=pad_value, max_length=max_length
    )

    X = np.stack([arr.T for arr in series_list], axis=0)
    y = np.array(labels)

    return X, y, paths, class_names, target_length


def collect_files_recursive(data_dir, extensions=(".xls", ".xlsx")):
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {data_dir}")
    files = []
    for ext in extensions:
        files.extend(sorted(data_dir.rglob(f"*{ext}")))
    return [str(path) for path in files]


def load_files(
    file_paths,
    pad_mode="pad",
    pad_value=0.0,
    target_length=None,
    nan_strategy="zero",
    show_progress=True,
    return_lengths=False,
):
    """Load unlabeled files and apply padding/truncation.

    Args:
        file_paths (list[str] or list[Path]): Files to load.
        pad_mode (str): Padding/truncation mode.
        pad_value (float): Padding value.
        target_length (int or None): Fixed length override.
        nan_strategy (str): NaN handling strategy.
        show_progress (bool): Whether to show a progress indicator.
        return_lengths (bool): Whether to return original lengths.
    """
    if not file_paths:
        raise ValueError("No input files provided")

    series_list = []
    paths = []
    lengths = []

    items_iter = file_paths
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm
        except Exception:
            pass
        else:
            items_iter = tqdm(file_paths, desc="Loading files", unit="file")
            use_tqdm = True

    total = len(file_paths)
    for idx, file_path in enumerate(items_iter, start=1):
        if show_progress and not use_tqdm:
            print(f"\rLoading files {idx}/{total}", end="", flush=True)
        arr = _read_excel(file_path, nan_strategy=nan_strategy)
        series_list.append(arr)
        paths.append(str(file_path))
        lengths.append(arr.shape[0])
    if show_progress and not use_tqdm:
        print()

    series_list, target_length = _pad_or_truncate(
        series_list, pad_mode=pad_mode, pad_value=pad_value, max_length=target_length
    )
    X = np.stack([arr.T for arr in series_list], axis=0)

    if return_lengths:
        return X, paths, target_length, lengths
    return X, paths, target_length


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def load_json(path):
    return json.loads(Path(path).read_text())
