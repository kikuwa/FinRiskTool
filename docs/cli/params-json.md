# `--params-json` 超参文件说明

通过 `--params-json PATH` 传入 JSON 对象，**合并覆盖**内置默认值。未出现的键保持默认。

---

## PU（`pu train` / `pu autoresearch start`）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `n_estimators` | int | `200` | Bagging 子模型个数 |
| `imbalance_ratio` | float | `0.2` | 每次迭代 U 采样量 = P × ratio |
| `verbosity` | int | `-1` | LightGBM 日志级别 |
| `learning_rate` | float | `0.05` | 学习率 |
| `num_leaves` | int | `20` | 叶子数 |
| `n_jobs` | int | `-1` | 并行线程，-1 为全部 |
| `scale_pos_weight` | float | `2` | 正样本权重 |
| `max_depth` | int | `4` | 树深度 |
| `min_child_samples` | int | `50` | 叶节点最小样本 |
| `subsample` | float | `0.7` | 行采样 |
| `colsample_bytree` | float | `0.7` | 列采样 |
| `num_boost_round` | int | `1200` | 每子模型 boosting 轮数 |

CLI 另有 `--num-boost-round`（默认 `1200`），传入训练入口并**优先于** JSON 中的 `num_boost_round`。

**示例** `config/pu_params.json`：

```json
{
  "n_estimators": 200,
  "imbalance_ratio": 0.2,
  "learning_rate": 0.05,
  "max_depth": 4,
  "num_boost_round": 1200
}
```

```bash
python cli.py pu train --params-json config/pu_params.json --rate 10
```

---

## 特征工程（`fe run`）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `estimated_positive_rate` | float | `0.1` | PN 构表正样本比例；CLI `--rate` 也会覆盖 |
| `mi_feature_threshold` | int | `100` | MI 阶段保留特征数上限 |
| `stability_top_k` | int | `50` | LGB 稳定性每轮 Top-K |
| `stability_min_hits` | int | `2` | 至少命中次数才入候选 |
| `stability_rus_ratio` | float | `0.15` | 稳定性阶段 RUS 比例 |
| `stability_lgb_n_estimators` | int | `200` | 稳定性 LGB 树数 |
| `stability_lgb_learning_rate` | float | `0.05` | 稳定性 LGB 学习率 |
| `stability_lgb_max_depth` | int | `6` | 稳定性 LGB 深度 |
| `stability_lgb_num_leaves` | int | `31` | 稳定性 LGB 叶子数 |

**示例** `config/fe_params.json`：

```json
{
  "mi_feature_threshold": 80,
  "stability_top_k": 50,
  "stability_min_hits": 2,
  "stability_rus_ratio": 0.15
}
```

---

## MLBase（`mlbase compare` / `train` / `test` / `autoresearch start`）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `recall_target` | float | `0.5` | 验证集召回下限；CLI `--recall-target` 可覆盖 |
| `learning_rate` | float | `0.05` | 学习率 |
| `n_estimators` | int | `150` | 树数量 |
| `max_depth` | int | `6` | 树深度 |
| `min_child_samples` | int | `50` | 叶节点最小样本 |
| `subsample` | float | `0.8` | 行采样 |
| `reg_alpha` | float | `0.1` | L1 正则 |
| `reg_lambda` | float | `0.1` | L2 正则 |

**示例** `config/ml_params.json`：

```json
{
  "recall_target": 0.6,
  "learning_rate": 0.05,
  "n_estimators": 150,
  "max_depth": 6,
  "min_child_samples": 50,
  "subsample": 0.8,
  "reg_alpha": 0.1,
  "reg_lambda": 0.1
}
```

```bash
python cli.py mlbase compare --params-json config/ml_params.json --recall-target 0.6
```

> 若同时指定 `--recall-target` 与 JSON 中的 `recall_target`，**命令行优先**。

---

## autoresearch 专用说明

- `--params-json` 提供 **第一轮初始超参**；后续轮次由 LLM 自动调整。
- `--rate`（PU / FE）与 JSON 内 `estimated_positive_rate`：FE 的 `--rate` 在 pipeline 入口覆盖；PU 的 `--rate` 为业务预估比例，与 PU JSON 无关。
- MLBase autoresearch 中 LLM **不可修改** `recall_target`（仅用户通过 CLI / JSON 固定）。
