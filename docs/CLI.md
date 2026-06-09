# finRiskTool CLI 使用指南

命令行入口（**Linux / macOS / Windows 通用**）：

```bash
python cli.py <子命令> [选项]
```

在项目根目录执行。CLI 与 GUI 共用服务层与 `data/` 输出目录。

## 文档索引

| 文档 | 内容 |
|------|------|
| [LOCAL_LLM.md](LOCAL_LLM.md) | **本地 / 内网 OpenAI 兼容模型**配置与需修改的文件清单 |
| [cli/examples.md](cli/examples.md) | 跨平台环境准备、端到端流水线（Bash / PowerShell / CMD） |
| [cli/parameters.md](cli/parameters.md) | **全部命令的可选参数**（含默认值与说明） |
| [cli/params-json.md](cli/params-json.md) | `--params-json` 文件字段、默认值与示例 |

## 快速开始

### 1. 激活虚拟环境

**Linux / macOS**

```bash
cd /path/to/finRiskTool
source .venv/bin/activate
export OPENAI_API_KEY=sk-your-key   # autoresearch 时需要
```

**Windows PowerShell**

```powershell
cd D:\path\to\finRiskTool
.\.venv\Scripts\Activate.ps1
$env:OPENAI_API_KEY = "sk-your-key"
```

**Windows CMD**

```cmd
cd D:\path\to\finRiskTool
.venv\Scripts\activate.bat
set OPENAI_API_KEY=sk-your-key
```

### 2. 查看帮助

```bash
python cli.py --help
python cli.py pu train --help
python cli.py pu autoresearch start --help
```

### 3. 最小流水线

将 `LABEL` 换成你的标签列名（如 `RevolverStatus`）：

```bash
# 方式 A：系统切分全量数据
python cli.py dataset split --label-col LABEL --test-size 0.3
python cli.py pu train --dataset split --label-col LABEL --rate 10 --timeout-minutes 20

# 方式 B：自定义训练/测试集路径（可跳过 dataset split）
python cli.py pu train --dataset split --label-col LABEL \
  --train-path /path/to/train.csv --test-path /path/to/test.csv --rate 10

python cli.py fe run --dataset split --label-col LABEL --rate 10
python cli.py fe confirm --default
python cli.py mlbase compare --dataset split --recall-target 0.6
python cli.py mlbase test --variant top_features --dataset split
```

加 `--json` 可输出结构化 JSON。

## 命令树

```
python cli.py
├── dataset
│   ├── split
│   └── analyze
├── pu
│   ├── train
│   └── autoresearch
│       ├── start
│       ├── stop
│       └── status
├── fe
│   ├── run
│   ├── confirm
│   └── rfecv
└── mlbase
    ├── compare
    ├── train
    ├── test
    └── autoresearch
        ├── start
        ├── stop
        └── status
```

## 全局与环境

### 多数命令共用的选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `--project-root` | 当前工作目录 | 项目根路径 |
| `--label-col` | 读 `dataset_preferences.json` | 标签列名 |
| `--json` | 关闭 | JSON 格式输出 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM API Key（可留空；本地模型见 [LOCAL_LLM.md](LOCAL_LLM.md)） |
| `OPENAI_BASE_URL` | API Base，如 `https://api.deepseek.com/v1` 或 `http://127.0.0.1:8000/v1` |
| `OPENAI_SSL_VERIFY` | 设为 `false` 跳过 SSL 校验 |
| `PU_PARAM_LLM_TIMEOUT` | LLM HTTP 超时（秒），默认 `180` |
| `PU_PARAM_LLM_MODEL` | PU autoresearch 模型，默认 `gpt-4o-mini` |
| `ML_PARAM_LLM_MODEL` | MLBase autoresearch 模型，默认 `gpt-4o-mini` |
| `FINRISK_CLI` | CLI 自动设为 `1`，避免 Flask 启动时清空 `data/results` |

### 数据目录

| 路径 | 说明 |
|------|------|
| `data/uploads/full_dataset.csv` | 全量数据 |
| `data/uploads/train_dataset.csv` | 训练集 |
| `data/uploads/test_dataset.csv` | 测试集 |
| `data/results/pu_learning/` | PU 预测、`pu_train.tsv` |
| `data/results/feature_selection/` | 特征候选、`top_features.csv`、ML 结果 |
| `data/results/dataset_preferences.json` | 标签列、切分比例 |

**推荐顺序**：`dataset split` → `pu train` → `fe run` → `fe confirm` → `mlbase compare` → `mlbase test`

## 与 GUI 对照

| GUI | CLI |
|-----|-----|
| 数据集管理 → 切分 | `dataset split` |
| PU Learning → 运行模型 | `pu train` |
| PU Learning → autoresearch | `pu autoresearch start --wait` |
| 特征工程 → 特征筛选 | `fe run` |
| 特征工程 → 确认特征 | `fe confirm --default` |
| 特征工程 → ML 对比 | `mlbase compare` |
| 特征工程 → Test 评估 | `mlbase test` |
| 特征工程 → ML autoresearch | `mlbase autoresearch start --wait` |

## 超时说明（autoresearch）

- **单轮训练**：由 `--timeout-minutes` / `--timeout-seconds` 控制；PU 默认 **10 分钟**，MLBase 默认 **20 分钟**。
- **LLM 调用**：HTTP 超时默认 **180 秒**（`PU_PARAM_LLM_TIMEOUT`）。
- **单轮总时长**、**整场 autoresearch 总时长**：无单独上限；靠用户停止或 `--max-invalid` 收敛。

## 常见问题

**`未找到 train_predictions.csv`** — 先运行 `pu train`。

**`未找到 top_features.csv`** — 先运行 `fe confirm`。

**autoresearch 停止** — 另开终端 `pu autoresearch stop`；`--wait` 模式下 `Ctrl+C` 也会发停止。

**CLI 与 GUI 混用** — 可以，勿同时跑两套 autoresearch。

详细参数见 [cli/parameters.md](cli/parameters.md)。跨平台完整示例见 [cli/examples.md](cli/examples.md)。
