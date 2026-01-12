# 雷达 LITETime 分类（鸟 vs 无人机）

本目录提供一个最小可用的训练/评估流程，用于识别 Excel 轨迹数据中的鸟与无人机。
使用 `aeon` 的 `LITETimeClassifier`，数据目录结构如下：

```
mydataset/radar_augv3/
  bird/
    *.xls
  uav/
    *.xls
```

每个 `.xls` 文件视为一个样本。数值列会作为通道（非数值列自动忽略），每一行代表一个时间步。
如果文件长度不一致，可通过参数进行补齐或截断到固定长度。

## 依赖与安装

- Python 依赖：`aeon`, `scikit-learn`, `tensorflow`, `joblib`, `pandas`, `xlrd`
- 示例安装（仓库内执行）：

```
pip install --editable .[dev]
pip install tensorflow pandas xlrd
```

> 说明：如果使用 `.xlsx`，需要 `openpyxl`（`pandas` 会自动调用）。

## 训练

```
python radar_litetime/train.py \
  --data-dir mydataset/radar_augv3 \
  --output-dir radar_litetime/output \
  --classes bird,uav \
  --extensions .xls \
  --pad-mode pad \
  --pad-value 0 \
  --nan-strategy zero \
  --save-best-model \
  --n-epochs 200 \
  --batch-size 64
```

输出文件：
- `radar_litetime/output/*.keras`（保存的模型权重）
- `radar_litetime/output/split.json`
- `radar_litetime/output/meta.json`

## 测试

```
python radar_litetime/test.py \
  --data-dir mydataset/radar_augv3 \
  --split-file radar_litetime/output/split.json \
  --meta-file radar_litetime/output/meta.json
```

如需手动指定模型文件（覆盖 meta.json）：

```
python radar_litetime/test.py \
  --data-dir mydataset/radar_augv3 \
  --model-dir radar_litetime/output \
  --model-files litetime_best0.keras,litetime_best1.keras,litetime_best2.keras
```

## 文件夹批量预测（递归）

对指定目录递归遍历 `.xls` 文件，输出每个文件的预测类别，并在最后统计各类别数量。

```
python radar_litetime/predict_folder.py \
  --input-dir /path/to/folder \
  --model-dir radar_litetime/output \
  --meta-file radar_litetime/output/meta.json
```

如需手动指定长度与补全策略（确保特征与训练一致）：

```
python radar_litetime/predict_folder.py \
  --input-dir /path/to/folder \
  --model-files litetime_best0.keras,litetime_best1.keras \
  --classes bird,uav \
  --length 400 \
  --pad-mode pad \
  --pad-value 0 \
  --nan-strategy zero
```

输出示例（每行：文件路径 + 预测类别）：

```
/path/to/folder/a.xls    bird
/path/to/folder/b.xls    uav
Class counts:
bird    123
uav     98
```

## 说明

- `.xls` 中存在多列数值时，会被视为多通道输入。
- 若要求严格等长输入，训练时设置 `--pad-mode error`。
- 若需固定长度，训练时设置 `--max-length`，测试会从 `meta.json` 复用该长度。
 - 批量预测默认从 `meta.json` 读取 `target_length/pad_mode/pad_value/nan_strategy`，可通过参数覆盖。
 - 长度不足时默认补齐（`pad`），长度超出则截断到目标长度；若使用 `error` 将直接报错。

## 参数调优建议（针对当前数据）

已知条件：每条航迹报文长度约 20 个点，默认使用 14 个特征（通道），单类约 13000 条报文。

- `--n-epochs`：深度模型训练轮数。建议先用 100~200 快速验证，再逐步增加到 500+。
- `--n-classifiers`：LITETime 集成子模型数，默认 5。更高可提升稳定性，但训练更慢。
- `--kernel-size`：卷积核长度。序列较短时可适当减小（例如 16~32）以避免过宽感受野。
- `--n-filters`：每层滤波器数量。资源允许时可提升到 64 以增强表达能力。
- `--batch-size`：与显存/内存相关，推荐从 32/64 起步。

推荐起步组合：`--n-epochs 200 --n-classifiers 5 --kernel-size 32 --n-filters 32 --batch-size 64`。
