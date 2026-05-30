# autoresearch

这是一个让LLM自己做研究的实验。

## Setup

要设置新实验，请与用户合作：

1. **仔细阅读范围内文件**：仓库规模很小。请阅读这些文件以获取完整背景：
   - `dataset_analysis_report.json` — 数据的信息
   - `PU_bagging.py` — 算法的详细介绍
   - `pu_train.tsv` — 每一轮使用的参数，实验的结果以及训练时长。

## Experimentation

**你可以做的事情**：修改算法里涉及的参数。

**你不可以做的事情**：修改算法逻辑、修改实验结果和数据信息等。

**约束项**：实验时长是约束项，如果实验运行超过5min，将会被kill。

**目标**：很简单，在预估正样本比例的前提下，获取更高的F1分数。



## Output format

Once the script finishes it prints a summary like this:

```
PU_PARAM_KEYS = (
    'n_estimators',
    'imbalance_ratio',
    'verbosity',
    'learning_rate',
    'num_leaves',
    'n_jobs',
    'scale_pos_weight',
    'max_depth',
    'min_child_samples',
    'subsample',
    'colsample_bytree',
    'num_boost_round',
)
必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{PU_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，简要说明各参数选择依据>"
}}
params 中必须且仅包含以下键：n_estimators, imbalance_ratio, verbosity, learning_rate, num_leaves, n_jobs, scale_pos_weight, max_depth, min_child_samples, subsample, colsample_bytree, num_boost_round，
检查对于的<number>是否按照预期更新
```

## The experiment loop

1. 当**近3次迭代优化都低于最优**，使用更激进的策略优化参数。
2. 当实验**次数超过10次**，先总结历史的经验，再思考下一步的优化策略。
3. 一旦实验开始，不要停下来问人类是否应该继续。这个人可能已经睡着了，或者已经离开电脑，期望你无限期地工作，直到你被手动停止。
