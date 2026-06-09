# CLI 跨平台示例

本文所有命令在项目根目录执行，形式均为：

```bash
python cli.py <命令> [选项]
```

将 `LABEL` 替换为实际标签列（如 `RevolverStatus`）。

---

## 环境准备

### Linux / macOS (Bash)

```bash
cd /path/to/finRiskTool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="sk-your-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
# 公司代理 SSL 问题时：
# export OPENAI_SSL_VERIFY=false
```

### Windows PowerShell

```powershell
cd D:\path\to\finRiskTool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:OPENAI_API_KEY = "sk-your-key"
$env:OPENAI_BASE_URL = "https://api.deepseek.com/v1"
# $env:OPENAI_SSL_VERIFY = "false"
```

### Windows CMD

```cmd
cd D:\path\to\finRiskTool
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt

set OPENAI_API_KEY=sk-your-key
set OPENAI_BASE_URL=https://api.deepseek.com/v1
```

---

## 端到端流水线

### Bash (Linux / macOS)

```bash
LABEL=RevolverStatus

python cli.py dataset split --label-col "$LABEL" --test-size 0.3
python cli.py dataset analyze --label-col "$LABEL"

python cli.py pu train \
  --dataset split \
  --label-col "$LABEL" \
  --rate 10 \
  --timeout-minutes 20

python cli.py fe run --dataset split --label-col "$LABEL" --rate 10
python cli.py fe confirm --default

python cli.py mlbase compare --recall-target 0.6
python cli.py mlbase test --variant top_features
```

### PowerShell (Windows)

```powershell
$LABEL = "RevolverStatus"

python cli.py dataset split --label-col $LABEL --test-size 0.3
python cli.py dataset analyze --label-col $LABEL

python cli.py pu train `
  --dataset split `
  --label-col $LABEL `
  --rate 10 `
  --timeout-minutes 20

python cli.py fe run --dataset split --label-col $LABEL --rate 10
python cli.py fe confirm --default

python cli.py mlbase compare --recall-target 0.6
python cli.py mlbase test --variant top_features
```

### CMD (Windows)

```cmd
set LABEL=RevolverStatus

python cli.py dataset split --label-col %LABEL% --test-size 0.3
python cli.py pu train --dataset split --label-col %LABEL% --rate 10 --timeout-minutes 20
python cli.py fe run --dataset split --label-col %LABEL% --rate 10
python cli.py fe confirm --default
python cli.py mlbase compare --recall-target 0.6
python cli.py mlbase test --variant top_features
```

---

## autoresearch

### PU — 前台等待（推荐）

**Bash**

```bash
python cli.py pu autoresearch start \
  --dataset split \
  --label-col RevolverStatus \
  --rate 10 \
  --timeout-minutes 20 \
  --max-invalid 3 \
  --wait
```

**PowerShell**

```powershell
python cli.py pu autoresearch start `
  --dataset split `
  --label-col RevolverStatus `
  --rate 10 `
  --timeout-minutes 20 `
  --max-invalid 3 `
  --wait
```

### PU — 后台启动 + 另终端查询/停止

```bash
# 终端 1（不加 --wait 即为后台）
python cli.py pu autoresearch start --api-key "$OPENAI_API_KEY"

# 终端 2
python cli.py pu autoresearch status
python cli.py pu autoresearch stop
```

### MLBase autoresearch

```bash
python cli.py mlbase autoresearch start \
  --variant top_features \
  --dataset split \
  --recall-target 0.6 \
  --timeout-minutes 20 \
  --max-invalid 3 \
  --wait
```

`Ctrl+C` 在 `--wait` 模式下会发送停止请求。

---

## 使用超参 JSON

**Bash**

```bash
python cli.py pu train --params-json config/pu_params.json --rate 10
python cli.py fe run --params-json config/fe_params.json
python cli.py mlbase compare --params-json config/ml_params.json --recall-target 0.6
```

**PowerShell**

```powershell
python cli.py pu train --params-json config\pu_params.json --rate 10
```

JSON 字段说明见 [params-json.md](params-json.md)。

---

## JSON 输出（脚本集成）

```bash
python cli.py pu train --label-col RevolverStatus --rate 10 --json > result.json
python cli.py pu autoresearch status --json
```

PowerShell 重定向：

```powershell
python cli.py pu autoresearch status --json | Out-File -Encoding utf8 status.json
```

---

## 指定项目根目录

在任意目录调用时，可显式指定 `--project-root`：

```bash
python cli.py --help   # 不支持全局 project-root，需在子命令上指定

python cli.py pu train \
  --project-root /path/to/finRiskTool \
  --label-col RevolverStatus \
  --rate 10
```

---

## 可执行脚本（可选）

仓库提供示例脚本，内容与上文等价：

- Linux / macOS：`scripts/run_pipeline.sh`
- Windows：`scripts/run_pipeline.ps1`

用法见脚本内注释。
