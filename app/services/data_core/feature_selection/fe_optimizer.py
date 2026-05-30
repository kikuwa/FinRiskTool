"""特征选择流水线 LLM 参数推荐。"""
import os
import sys
from typing import Any, Dict, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.data_core.pu.param_optimizer import (
    _create_openai_client,
    _extract_json_object,
    _load_dataset_report,
    _normalize_base_url,
    format_llm_connection_error,
)

FE_USER_PARAM_KEYS = (
    'estimated_positive_rate',
    'mi_feature_threshold',
    'stability_top_k',
    'stability_min_hits',
)

FE_AI_PARAM_KEYS = (
    'stability_rus_ratio',
    'stability_lgb_n_estimators',
    'stability_lgb_learning_rate',
    'stability_lgb_max_depth',
    'stability_lgb_num_leaves',
)

FE_PARAM_KEYS = FE_USER_PARAM_KEYS + FE_AI_PARAM_KEYS

DEFAULT_FE_PARAMS: Dict[str, float] = {
    'estimated_positive_rate': 0.1,
    'mi_feature_threshold': 100,
    'stability_top_k': 50,
    'stability_min_hits': 2,
    'stability_rus_ratio': 0.15,
    'stability_lgb_n_estimators': 200,
    'stability_lgb_learning_rate': 0.05,
    'stability_lgb_max_depth': 6,
    'stability_lgb_num_leaves': 31,
}


def _load_fe_algorithm_source(project_root: str) -> str:
    path = os.path.join(os.path.dirname(__file__), 'pipeline.py')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _clamp_fe_params(params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_FE_PARAMS)
    for key in FE_PARAM_KEYS:
        if key not in params:
            continue
        val = params[key]
        if key == 'estimated_positive_rate':
            merged[key] = float(max(0.01, min(0.5, float(val))))
        elif key == 'mi_feature_threshold':
            merged[key] = int(max(10, min(500, float(val))))
        elif key == 'stability_top_k':
            merged[key] = int(max(10, min(200, float(val))))
        elif key == 'stability_min_hits':
            merged[key] = int(max(1, min(3, float(val))))
        elif key == 'stability_rus_ratio':
            merged[key] = float(max(0.05, min(0.5, float(val))))
        elif key == 'stability_lgb_n_estimators':
            merged[key] = int(max(50, min(500, float(val))))
        elif key == 'stability_lgb_learning_rate':
            merged[key] = float(max(0.001, min(0.3, float(val))))
        elif key == 'stability_lgb_max_depth':
            merged[key] = int(max(2, min(20, float(val))))
        elif key == 'stability_lgb_num_leaves':
            merged[key] = int(max(8, min(255, float(val))))
    return merged


def _clamp_fe_ai_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """仅校验 AI 可调键（用户设定由界面固定）。"""
    full = _clamp_fe_params(params)
    return {key: full[key] for key in FE_AI_PARAM_KEYS}


def _build_fe_system_prompt() -> str:
    keys_desc = ', '.join(FE_AI_PARAM_KEYS)
    user_keys_desc = ', '.join(FE_USER_PARAM_KEYS)
    return f"""你是金融风控与机器学习专家，熟悉 PU Learning、特征选择与 LightGBM。

请根据数据集分析报告与特征选择算法源码，推荐稳定性实验中的模型超参数。

以下参数由用户在界面固定，**禁止**在 params 中输出或修改：{user_keys_desc}

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{FE_AI_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，简要说明各参数选择依据>"
}}

params 中必须且仅包含以下键：{keys_desc}
不可修改算法逻辑，仅调整上述模型超参数。"""


def _build_fe_user_prompt(dataset_report: str, algorithm_source: str) -> str:
    return (
        f'## dataset_analysis_report.json\n{dataset_report}\n\n'
        f'## ensemble_feature_selection.py（特征选择算法）\n{algorithm_source}\n\n'
        '请仅推荐稳定性实验的模型超参数（RUS 与 LGB 相关项）。'
        'estimated_positive_rate、mi_feature_threshold、stability_top_k、stability_min_hits 由用户设定，勿输出。'
    )


def optimize_fe_params_with_llm(
    project_root: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = 'gpt-4o-mini',
) -> Dict[str, Any]:
    dataset_report = _load_dataset_report(project_root)
    algorithm_source = _load_fe_algorithm_source(project_root)

    client = _create_openai_client(api_key, base_url)
    resolved_base = _normalize_base_url(base_url) or os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    )
    print(
        f'[FE Param Optimizer] model={model}, base={resolved_base}',
        file=sys.stderr,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': _build_fe_system_prompt()},
                {
                    'role': 'user',
                    'content': _build_fe_user_prompt(dataset_report, algorithm_source),
                },
            ],
            temperature=0.3,
            max_tokens=2000,
        )
    except (APIConnectionError, APITimeoutError) as e:
        raise ConnectionError(format_llm_connection_error(e)) from e
    except APIStatusError as e:
        raise RuntimeError(f'API 返回错误 {e.status_code}: {e.message}') from e

    raw_content = (response.choices[0].message.content or '').strip()
    parsed = _extract_json_object(raw_content)
    params_raw = parsed.get('params') or parsed
    if not isinstance(params_raw, dict):
        raise ValueError('大模型返回的 JSON 缺少 params 字段')

    return {
        'params': _clamp_fe_ai_params(params_raw),
        'reasoning': parsed.get('reasoning', ''),
        'llm_response': raw_content,
        'model': model,
    }
