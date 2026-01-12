# 雷达 MultiROCKET 分类（鸟 vs 无人机）

本目录提供一个最小可用的训练/评估流程，用于识别 Excel 轨迹数据中的鸟与无人机。
使用 `aeon` 的 `MultiRocketClassifier`，数据目录结构如下：

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

- Python 依赖：`aeon`, `scikit-learn`, `joblib`, `pandas`, `xlrd`
- 示例安装（仓库内执行）：

```
pip install --editable .[dev]
pip install pandas xlrd
```

## 训练

```
python radar_multirocket/train.py \
  --data-dir mydataset/radar_augv3 \
  --output-dir radar_multirocket/output \
  --classes bird,uav \
  --extensions .xls \
  --pad-mode pad \
  --pad-value 0 \
  --nan-strategy zero
```

输出文件：
- `radar_multirocket/output/model.joblib`
- `radar_multirocket/output/split.json`
- `radar_multirocket/output/meta.json`

## 测试

```
python radar_multirocket/test.py \
  --data-dir mydataset/radar_augv3 \
  --model-path radar_multirocket/output/model.joblib \
  --split-file radar_multirocket/output/split.json
```

## 说明

- `.xls` 中存在多列数值时，会被视为多通道输入。
- 若要求严格等长输入，训练时设置 `--pad-mode error`。
- 若需固定长度，训练时设置 `--max-length`，测试会从 `meta.json` 复用该长度。

## 参数调优建议（针对当前数据）

已知条件：每条航迹报文长度约 20 个点，默认使用 14 个特征（通道），单类约 13000 条报文。

- `--n-kernels`（核数量）：主要影响特征维度与训练时间。样本量较多时可提高到 10000~20000；若训练过慢或内存不足，可先降到 2000~5000 做验证。
- `--n-features-per-kernel`（每核特征数）：MultiROCKET 默认 4，属于稳妥起点。你的序列较短（20 点）且通道数 14，通常保持 4 即可；如果需要更强表达力可尝试 6~8，但训练时间与内存会增加。
- `--max-dilations-per-kernel`（每核最大膨胀数）：控制不同尺度的感受野数量。序列较短时，过大的膨胀数收益有限且会增加特征计算量。建议从默认 32 起步；若训练太慢或收益不明显，可降到 16；若需要更强的多尺度表达且资源允许，可尝试 48 或 64。

推荐起步组合：`--n-kernels 10000 --n-features-per-kernel 4`。如需更快迭代，先用 `--n-kernels 3000` 做小规模对比，再逐步增大。
