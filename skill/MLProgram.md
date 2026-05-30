# autoresearch

这是一个让 LLM 自己做 MLBase 监督学习超参研究的实验。

## Setup

要设置新实验，请与用户合作：

1. **仔细阅读范围内文件**：请阅读这些文件以获取完整背景：
   - `dataset_analysis_report.json` — 数据的信息
   - `app/services/data_core/mlbase/core.py`（MLBaseModel）— 单次 LightGBM 训练与验证集阈值调优逻辑
   - `ml_train.tsv` — 每一轮使用的参数、特征方案（variant）、验证集最佳阈值、召回、精度与训练时长
   - `top_features.csv` — 当 variant 为 `top_features` 时已确认特征列表
   - `skill/MLProgram.md` — 本规程

用户会在界面选定 **variant**（`top_features` 或 `full_features`），autoresearch 仅在该方案下迭代，**不可自行更换特征列表**。

启动 autoresearch 时，若已有「运行 ML 对比」结果，系统会自动将其写入 `ml_train.tsv`，`时间` 列为 `baseline`，作为首轮 LLM 参考。

用户消息中的 **当前特征方案** 段即为模型实际训练用的特征表（`top_features` 时为完整 `top_features.csv`；`full_features` 时为训练集除标签外的全部列）。**必须以该表为准**；若提示「特征集已变更」，说明相对上次 ML 对比的特征已变化，勿假设历史实验与当前特征一致。

## Experimentation

**你可以做的事情**：修改 `MLBASE_PARAM_KEYS` 中的超参数。

**你不可以做的事情**：修改算法逻辑、修改实验结果和数据信息、更换 variant 或特征列。

**约束项**：实验时长是约束项。若单次训练超过界面配置的「训练超时（分钟）」，将被 kill 并记入 `ml_train.tsv` 的 `时间` 列为 `timeout`。

**目标**：在 **用户设定的 `recall_target` 约束下**（该值由界面固定，autoresearch 不得修改），通过验证集阈值搜索使 **验证集 precision 尽可能高**。

**AI 可调键**（仅以下键可出现在 params JSON 中）：learning_rate, n_estimators, max_depth, min_child_samples, subsample, reg_alpha, reg_lambda。

**用户固定键**：recall_target（阈值搜索的召回下限，由界面设定）。

## ml_train.tsv 列说明

表头（`|` 分隔）：

```
variant|参数|验证阈值|验证召回|验证精度|时间
```

- **variant**：`top_features` 或 `full_features`
- **参数**：分号分隔的 `key=value`，键为 MLBASE 超参
- **验证阈值 / 验证召回 / 验证精度**：该轮验证集最优阈值及对应指标
- **时间**：秒（小数）；失败为 `NAN`；超时为 `timeout`；启动 autoresearch 时从 ML 对比写入的基线为 `baseline`

## Output format

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：

```
MLBASE_PARAM_KEYS = (
    'recall_target',
    'learning_rate',
    'n_estimators',
    'max_depth',
    'min_child_samples',
    'subsample',
    'reg_alpha',
    'reg_lambda',
)
{
  "params": {
    "recall_target": <number>,
    ...
  },
  "reasoning": "<中文，简要说明各参数选择依据>"
}
```

params 中必须且仅包含以下 AI 可调键：learning_rate, n_estimators, max_depth, min_child_samples, subsample, reg_alpha, reg_lambda。勿输出 recall_target。

## The experiment loop

1. 当**近 3 次迭代验证精度都未超过历史最优**，使用更激进的策略优化参数。
2. 当实验**次数超过 10 次**，先总结历史的经验，再思考下一步的优化策略。
3. 一旦实验开始，不要停下来问人类是否应该继续。期望你持续工作，直到被手动停止或达到最大无效迭代次数。
