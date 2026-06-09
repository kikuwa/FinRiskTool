# CLI 参数参考

标记说明：

- **可选**：可省略，有默认值
- **必填**：必须提供（或满足互斥组之一）
- **环境变量**：可通过环境变量替代 CLI 选项

全局选项 `--project-root`、`--label-col`、`--json` 在下列各命令中均可使用（若该命令已注册）。

---

## 全局共用选项

| 选项 | 类型 | 默认 | 适用命令 | 说明 |
|------|------|------|----------|------|
| `--project-root` | 路径 | 当前目录 | 几乎全部 | 项目根目录 |
| `--label-col` | 字符串 | 读 `dataset_preferences.json`，否则 `label` | 几乎全部 | 标签列名 |
| `--json` | 开关 | 关 | 几乎全部 | 结果以 JSON 输出 |

---

## `dataset split`

分层切分 train/test，写入 `data/uploads/train_dataset.csv` 与 `test_dataset.csv`。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--input` | 路径 | `data/uploads/full_dataset.csv` | 全量 CSV 输入 |
| `--test-size` | 浮点 | `0.3` | 测试集比例 (0, 1) |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `dataset analyze`

生成 `data/results/dataset_analysis_report.json`。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--input` | 路径 | `data/uploads/full_dataset.csv` | 分析用 CSV |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `pu train`

单次 PU Bagging 训练。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset` | `split` \| `full` | `split` | `split` 用 train/test 文件；`full` 用全量再划分 |
| `--train-path` | 路径 | `data/uploads/train_dataset.csv` | 仅 `split`：自定义训练集 CSV |
| `--test-path` | 路径 | `data/uploads/test_dataset.csv` | 仅 `split`：自定义测试集 CSV |
| `--rate` | 浮点 | `10` | 预估正样本比例；`10` 表示 10% |
| `--timeout-minutes` | 浮点 | 未设 → **10 分钟** | 训练超时（分钟） |
| `--timeout-seconds` | 整数 | 未设 | 训练超时（秒）；优先于 minutes |
| `--params-json` | 路径 | 无 | PU 超参 JSON，见 [params-json.md](params-json.md) |
| `--num-boost-round` | 整数 | `1200` | LightGBM 每子模型 boosting 轮数 |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**输出**：`data/results/pu_learning/train_predictions.csv`、`test_predictions.csv`、`pu_train.tsv`

---

## `pu autoresearch start`

PU LLM 自动调参循环。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset` | `split` \| `full` | `split` | 数据集模式 |
| `--train-path` | 路径 | 见 `pu train` | 仅 `split`：自定义训练集 |
| `--test-path` | 路径 | 见 `pu train` | 仅 `split`：自定义测试集 |
| `--rate` | 浮点 | `10` | 预估正样本比例 |
| `--timeout-minutes` | 浮点 | 未设 → **10 分钟** | **每轮训练**超时 |
| `--timeout-seconds` | 整数 | 未设 | 每轮训练超时（秒） |
| `--api-key` | 字符串 | **`OPENAI_API_KEY`** | LLM API Key |
| `--base-url` | URL | **`OPENAI_BASE_URL`** | API Base |
| `--model` | 字符串 | **`PU_PARAM_LLM_MODEL`** 或 `gpt-4o-mini` | 模型名 |
| `--max-invalid` | 整数 | `3` | 连续无效轮次上限后自动停止 |
| `--params-json` | 路径 | 无 | 初始 PU 超参 JSON |
| `--num-boost-round` | 整数 | `1200` | boosting 轮数 |
| `--wait` | 开关 | 关 | 前台轮询日志直至结束 |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**LLM 超时**：由 `PU_PARAM_LLM_TIMEOUT` 控制（默认 180s），非 CLI 选项。

---

## `pu autoresearch stop` / `status`

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 仅影响输出上下文 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `fe run`

特征选择（PN + MI + LGB 稳定性）。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset` | `split` \| `full` | `split`（或读 `pu_session.json`） | 训练数据来源 |
| `--train-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：自定义训练集 |
| `--test-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：`fe run` 忽略；`rfecv`/`mlbase test` 可用 |
| `--rate` | 浮点 | `10` | 预估正样本比例（PN 构表） |
| `--params-json` | 路径 | 无 | 特征工程超参 JSON |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**前置条件**：`data/results/pu_learning/train_predictions.csv` 存在。

**输出**：`feature_candidates.csv`、`feature_selection_meta.json` 等。

---

## `fe confirm`

写入 `top_features.csv`。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--default` | 开关 | — | **与 `--features` 二选一**；使用候选表 `selected=true` 的行 |
| `--features` | 字符串 | — | **与 `--default` 二选一**；逗号分隔特征名 |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `fe rfecv`

RFECV 递归特征消除（全量优于 Top 时的重选流程）。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset` | `split` \| `full` | `split`（或读 session） | 训练数据 |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `mlbase compare`

全量特征 vs `top_features.csv` 对比训练。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--dataset` | `split` \| `full` | `split`（或读 session） | 训练数据 |
| `--train-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：自定义训练集 |
| `--test-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：本命令未使用 |
| `--recall-target` | 浮点 | `0.5` | 验证集召回下限（阈值搜索） |
| `--params-json` | 路径 | 无 | MLBase 超参 JSON |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**前置条件**：`top_features.csv` 已存在。

**输出**：`mlbase_comparison.json`

---

## `mlbase train`

单 variant 单次训练。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--variant` | `top_features` \| `full_features` | `top_features` | 特征方案 |
| `--dataset` | `split` \| `full` | `split`（或读 session） | 训练数据 |
| `--train-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：自定义训练集 |
| `--test-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：本命令未使用 |
| `--recall-target` | 浮点 | `0.5` | 召回下限 |
| `--timeout-minutes` | 浮点 | 未设 → **20 分钟** | 训练超时 |
| `--timeout-seconds` | 整数 | 未设 | 训练超时（秒） |
| `--params-json` | 路径 | 无 | MLBase 超参 JSON |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

---

## `mlbase test`

在 hold-out test 集上评估（使用 comparison 中的阈值）。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--variant` | `top_features` \| `full_features` | `top_features` | 使用哪套 comparison 结果 |
| `--dataset` | `split` \| `full` | `split` | **仅支持 split**（需 test CSV） |
| `--train-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：自定义训练集 |
| `--test-path` | 路径 | 默认或 `pu_session.json` | 仅 `split`：自定义测试集 |
| `--recall-target` | 浮点 | `0.5` | 写入 params-json 时覆盖 |
| `--params-json` | 路径 | 无 | 训练超参（影响 refit 模型） |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**前置条件**：`mlbase_comparison.json` 存在。

---

## `mlbase autoresearch start`

MLBase LLM 自动调参。

| 选项 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--variant` | `top_features` \| `full_features` | `top_features` | 固定特征方案（autoresearch 不可切换） |
| `--dataset` | `split` \| `full` | `split`（或读 session） | 训练数据 |
| `--recall-target` | 浮点 | `0.5` | 用户固定召回目标 |
| `--timeout-minutes` | 浮点 | 未设 → **20 分钟** | 每轮训练超时 |
| `--timeout-seconds` | 整数 | 未设 | 每轮训练超时（秒） |
| `--api-key` | 字符串 | **`OPENAI_API_KEY`** | LLM API Key |
| `--base-url` | URL | **`OPENAI_BASE_URL`** | API Base |
| `--model` | 字符串 | **`ML_PARAM_LLM_MODEL`** 或 `gpt-4o-mini` | 模型名 |
| `--max-invalid` | 整数 | `3` | 连续无效轮次上限 |
| `--params-json` | 路径 | 无 | 初始 MLBase 超参 JSON |
| `--wait` | 开关 | 关 | 前台等待 |
| `--project-root` | 路径 | 当前目录 | 项目根 |
| `--label-col` | 字符串 | 见全局 | 标签列 |
| `--json` | 开关 | 关 | JSON 输出 |

**前置条件**：`variant=top_features` 时需 `top_features.csv`。

---

## `mlbase autoresearch stop` / `status`

同 `pu autoresearch stop` / `status` 选项表。

---

## 超时默认值汇总

| 场景 | 默认超时 |
|------|----------|
| `pu train` / PU autoresearch 每轮训练 | 600s（10 分钟） |
| `mlbase train` / MLBase autoresearch 每轮训练 | 1200s（20 分钟） |
| LLM HTTP 请求 | 180s（`PU_PARAM_LLM_TIMEOUT`） |
| 最小可设训练超时 | 60s |
| 最大可设训练超时 | 86400s（24 小时） |
