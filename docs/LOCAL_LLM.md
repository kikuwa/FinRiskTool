# 本地部署大模型配置指南

本文说明如何将 finRiskTool 从云端 API（如 DeepSeek、OpenAI）切换为**本机或内网**部署的 **OpenAI 兼容**服务（vLLM、Ollama OpenAI 模式、LocalAI、XInference、LM Studio 等），以及若要在代码里改**默认** LLM 地址时需要动哪些文件。

---

## 1. 能力范围：哪些功能会调 LLM

| 模块 | 功能 | 调用方式 |
|------|------|----------|
| 数据工具 · PU Learning | 参数推荐、autoresearch | OpenAI SDK（`app/services/data_core/llm/client.py`） |
| 数据工具 · 特征工程 | FE 参数推荐、ML autoresearch | 同上 |
| 数据工具 · MLBase | ML 参数推荐、autoresearch | 同上 |
| Risk CoT · 推理 | 批量推理 | `requests` 直连 `.../chat/completions` |
| Risk CoT · 质检 | 模型打分 | `requests` 直连 `{api_base}/chat/completions` |
| Risk CoT · 数据合成 | Prompt 模板 LLM 优化 | OpenAI SDK |
| CLI | `pu autoresearch`、`mlbase autoresearch` | 与环境变量 / `--base-url` 一致 |

**不调用 LLM**：`dataset split`、`pu train`、`fe run`、`mlbase train/compare` 等纯机器学习流程，与模型 API 无关。

---

## 2. 推荐做法：不改代码（环境变量 + 页面配置）

多数场景**无需改仓库文件**，按下面配置即可。

### 2.1 环境变量（服务端 / CLI 共用）

在启动 `run.py` 或执行 CLI **之前**设置：

| 变量 | 说明 | 本地示例 |
|------|------|----------|
| `OPENAI_BASE_URL` | OpenAI 兼容 **Base**，建议以 `/v1` 结尾 | `http://127.0.0.1:8000/v1` |
| `OPENAI_API_KEY` | API Key；本地可不校验时可留空或任意占位 | `""` 或 `not-needed` |
| `PU_PARAM_LLM_MODEL` | PU / FE 参数优化、PU autoresearch 默认模型名 | `Qwen2.5-7B-Instruct` |
| `ML_PARAM_LLM_MODEL` | MLBase autoresearch 默认模型名 | 同上 |
| `PU_PARAM_LLM_TIMEOUT` | LLM HTTP 超时（秒），默认 `180` | 按需加大 |
| `OPENAI_SSL_VERIFY` | 内网 HTTP 或自签证书时可设 `false` | `false` |

**Linux / macOS**

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=
export PU_PARAM_LLM_MODEL=your-local-model
export ML_PARAM_LLM_MODEL=your-local-model
python run.py
```

**Windows PowerShell**

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY = ""
$env:PU_PARAM_LLM_MODEL = "your-local-model"
python run.py
```

**CLI autoresearch 示例**

```bash
python cli.py pu autoresearch start \
  --base-url "$OPENAI_BASE_URL" \
  --model "$PU_PARAM_LLM_MODEL" \
  --wait
```

未传 `--api-key` 时允许空 Key，请求仍会发出（`Authorization: Bearer `）。

### 2.2 Web 界面（浏览器 localStorage）

以下页面有 **API Key / API Base / 模型名** 输入框，配置会写入浏览器 `localStorage`（键名由 `app/static/js/llm_config.js` 定义），在 PU Learning、特征工程、CoT 推理之间**共用**：

- `app/templates/data_tool/pu_learning.html`
- `app/templates/data_tool/feature_engineering.html`
- `app/templates/risk_cot/inference.html`

填写说明：

- **API Base**：本地 vLLM 等填 `http://主机:端口/v1`（与 `OPENAI_BASE_URL` 一致）。
- **API Key**：服务不校验时可留空。
- **模型名**：与本地服务暴露的 `model`  id 一致（如 `Qwen2.5-7B-Instruct`）。

**注意**：`app/templates/risk_cot/inspector.html`（质检）目前**未**接入 `llm_config.js`，需在质检页单独填写 Key / Base / 模型。

### 2.3 Base URL 两种写法（避免 404）

项目里存在两条 HTTP 路径，填错地址会报 404：

| 路径 | 使用位置 | 应填写的 URL 形式 |
|------|----------|-------------------|
| OpenAI SDK | PU/FE/ML 参数优化、autoresearch、Prompt 优化 | `http://host:port/v1`（代码会自动规范化，也可写根地址，见 `normalize_base_url`） |
| 完整 chat 端点 | CoT **推理**（`inference_engine`） | `http://host:port/v1/chat/completions` 或根地址（页面经 `toChatCompletionsUrl` 转换） |
| 主机 + 路径拼接 | CoT **质检**（`model_inspector`） | 填 **不含** `/chat/completions` 的 host 根，如 `http://host:port/v1`；代码会拼 `{api_base}/chat/completions` |

本地 vLLM 常见启动：`--api-key optional` + `http://0.0.0.0:8000/v1`。

---

## 3. 修改代码时的文件清单（改默认 LLM / 文档）

若希望团队**克隆即用**本地地址、或统一文档说明，可按层级修改下列文件。**仅改环境变量时不必动这些文件。**

### 3.1 核心（优先了解）

| 文件 | 作用 |
|------|------|
| `app/services/data_core/llm/client.py` | **统一** OpenAI 客户端：`normalize_base_url`、`create_openai_client`、SSL、`PU_PARAM_LLM_TIMEOUT`、连接错误提示文案 |
| `app/static/js/llm_config.js` | 前端 Base URL 规范化（`/v1`、`/chat/completions`）、localStorage 键名 |

几乎所有「数据工具 + OpenAI SDK」路径最终都经过 `client.py`。

### 3.2 后端 · 路由（读请求体 + 环境变量）

| 文件 | 说明 |
|------|------|
| `app/routes/data_tool.py` | PU/ML autoresearch、`optimize_pu_params`、`optimize_fe_params`、`optimize_mlbase_params`：读取 `api_key` / `base_url` / `model`，回退 `OPENAI_*`、`PU_PARAM_LLM_MODEL`、`ML_PARAM_LLM_MODEL` |
| `app/routes/risk_cot/inference.py` | 推理任务：`base_url` 默认 `https://api.deepseek.com/chat/completions` |
| `app/routes/risk_cot/inspector.py` | 质检任务：透传 `api_key`、`api_base`、`model` |
| `app/routes/risk_cot/generator.py` | 模板 LLM 生成：透传 `api_key`、`base_url`、`model` |

改**服务端默认**时，重点改 `data_tool.py` 与各 `risk_cot/*.py` 中的默认值或统一改为读取环境变量。

### 3.3 后端 · 服务层（业务逻辑 / 兜底 URL）

| 文件 | 说明 |
|------|------|
| `app/services/data_core/pu/param_optimizer.py` | PU 参数推荐、autoresearch 调 LLM；兜底 `OPENAI_BASE_URL` 默认 `https://api.openai.com/v1` |
| `app/services/data_core/feature_selection/fe_optimizer.py` | FE 参数推荐；同上 |
| `app/services/data_core/mlbase/param_optimizer.py` | MLBase 参数推荐、autoresearch 建议；同上 |
| `app/services/data_core/pu/autoresearch.py` | 传递 `api_key` / `base_url` / `model`（一般无需改 URL 逻辑） |
| `app/services/data_core/mlbase/autoresearch.py` | 同上 |
| `app/services/risk_cot/inference_engine.py` | 推理：`requests.post(base_url, ...)`，默认 DeepSeek chat URL |
| `app/services/risk_cot/model_inspector.py` | 质检：类默认 `api_base="https://api.deepseek.com"`，请求 `{api_base}/chat/completions` |
| `app/services/risk_cot/prompt_engine.py` | Prompt 模板 LLM 优化（OpenAI SDK） |
| `app/services/risk_cot/inspector_engine.py` | 组装质检 config，传给 `model_inspector` |

若要把兜底从 OpenAI/DeepSeek 改为内网地址，需改 `*_optimizer.py` 中的 `os.environ.get('OPENAI_BASE_URL', '...')` 以及 `inference_engine.py`、`model_inspector.py` 的硬编码默认。

### 3.4 CLI

| 文件 | 说明 |
|------|------|
| `app/cli/commands/pu.py` | `pu autoresearch start`：`--api-key`、`--base-url`、`--model` 与环境变量 |
| `app/cli/commands/mlbase.py` | `mlbase autoresearch start`：同上，模型默认 `ML_PARAM_LLM_MODEL` |

### 3.5 前端模板（页面默认与占位符）

| 文件 | 说明 |
|------|------|
| `app/templates/data_tool/pu_learning.html` | 默认 `apiBase`、placeholder、帮助文案 |
| `app/templates/data_tool/feature_engineering.html` | 同上 |
| `app/templates/risk_cot/inference.html` | 默认 `apiBase` 为 DeepSeek chat URL；提交前 `toChatCompletionsUrl` |
| `app/templates/risk_cot/inspector.html` | 质检独立表单（无 `llm_config.js` 时可在此改 placeholder） |
| `app/templates/base.html` | 引入 `llm_config.js`（一般不改） |

修改 `apiBase: 'https://api.deepseek.com/...'` 等 **Alpine.js 初始值**即可改变首次打开页面的默认值。

### 3.6 文档（与实现对齐）

| 文件 | 说明 |
|------|------|
| `docs/CLI.md` | CLI 环境变量总表 |
| `docs/cli/parameters.md` | autoresearch 的 `--api-key` / `--base-url` / `--model` |
| `docs/cli/examples.md` | 示例中的 `OPENAI_*` 导出命令 |
| `docs/LOCAL_LLM.md` | 本文 |

---

## 4. 按场景的修改建议

### 场景 A：个人本机调试

1. 启动本地 OpenAI 兼容服务（记下端口与 model id）。
2. 设置 `OPENAI_BASE_URL`、`PU_PARAM_LLM_MODEL`（及可选 `OPENAI_API_KEY`）。
3. 浏览器打开 PU / 特征 / 推理页，确认 Base、模型名与本地一致；Key 可留空。

**不必改代码。**

### 场景 B：团队统一内网默认地址

1. 在部署脚本或 systemd/docker 中注入 `OPENAI_BASE_URL` 等环境变量（推荐）。
2. 若必须写死在仓库：改 `app/services/data_core/llm/client.py` 相关说明即可覆盖大部分 SDK 路径；另改 `inference_engine.py`、`model_inspector.py` 与三个 `*_optimizer.py` 的兜底 URL；前端三处模板的 `apiBase` 初始值。
3. 同步更新 `docs/cli/examples.md`、`docs/CLI.md` 中的示例 URL。

### 场景 C：仅 CLI / 自动化流水线

只配置环境变量 + `python cli.py ... autoresearch start --base-url ... --model ...`，无需改前端。

### 场景 D：自签证书或公司代理

- SDK 路径：`OPENAI_SSL_VERIFY=false`（见 `client.py`）。
- 代理：配置 `HTTP_PROXY` / `HTTPS_PROXY`，或对本地地址设置 `NO_PROXY=127.0.0.1,localhost`。

---

## 5. 自检清单

1. 用 `curl` 测本地服务是否可用：

   ```bash
   curl http://127.0.0.1:8000/v1/models
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"your-model","messages":[{"role":"user","content":"hi"}]}'
   ```

2. Web：PU Learning →「参数推荐」或 autoresearch，看日志 / 网络请求是否指向内网地址。

3. CLI：`python cli.py pu autoresearch status` 与带 `--base-url` 的 start。

4. CoT 推理：确认 `inference.html` 中 Base 为可访问的 chat 端点（或 `/v1` 由 JS 转换）。

5. 质检：在 inspector 页单独配置 Base（勿与推理页的 DeepSeek 默认混淆）。

---

## 6. 常见问题

**Q：留空 API Key 是否可行？**  
A：可以。当前版本已取消必填校验；本地服务忽略鉴权时 Key 留空即可，仍会发送 `Authorization: Bearer `。

**Q：为什么 PU 页填 `/v1`，推理页有时要写 `/chat/completions`？**  
A：数据工具走 OpenAI Python SDK（Base 为 `/v1`）；推理引擎用 `requests` 直接 POST 完整 chat URL。`llm_config.js` 的 `toOpenAiV1Base` / `toChatCompletionsUrl` 负责在页面间转换，共用 localStorage 时一般以 `/v1` 存储即可。

**Q：改一处能否全局生效？**  
A：环境变量 `OPENAI_BASE_URL` 对**服务端**未传 `base_url` 的请求生效；浏览器仍可能用 localStorage 覆盖。团队默认建议：**部署环境变量 + 改前端模板初始值** 双保险。

**Q：Ollama 示例**  
A：若启用 OpenAI 兼容：`OPENAI_BASE_URL=http://127.0.0.1:11434/v1`，`PU_PARAM_LLM_MODEL=llama3.2`（以 `ollama list` 为准）。

---

## 7. 相关文档

- [CLI 使用指南](CLI.md)
- [CLI 参数参考](cli/parameters.md)
- [CLI 示例](cli/examples.md)
