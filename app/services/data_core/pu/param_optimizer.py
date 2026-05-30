import os
import sys
from typing import Any, Dict, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.data_core.llm.client import (
    create_openai_client,
    extract_json_object,
    format_llm_connection_error,
    normalize_base_url,
    print_llm_response,
)
from app.services.data_core.llm.dataset_context import load_dataset_report

# 兼容 fe_param_optimizer / mlbase_param_optimizer 的旧 import
_create_openai_client = create_openai_client
_extract_json_object = extract_json_object
_normalize_base_url = normalize_base_url
_load_dataset_report = load_dataset_report


def _print_llm_response(content: str, log_folder: Optional[str] = None) -> None:
    print_llm_response(content, log_folder=log_folder, label='PU Param LLM Response')

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

DEFAULT_PU_PARAMS = {
    'n_estimators': 200,
    'imbalance_ratio': 0.2,
    'verbosity': -1,
    'learning_rate': 0.05,
    'num_leaves': 20,
    'n_jobs': -1,
    'scale_pos_weight': 2,
    'max_depth': 4,
    'min_child_samples': 50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'num_boost_round': 1200,
}


def _load_algorithm_source(project_root: str) -> str:
    algo_path = os.path.join(os.path.dirname(__file__), 'bagging.py')
    with open(algo_path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_pu_program(project_root: str) -> str:
    program_path = os.path.join(project_root, 'skill', 'puProgram.md')
    if not os.path.exists(program_path):
        raise FileNotFoundError('未找到 skill/puProgram.md')
    with open(program_path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_pu_train_log(project_root: str) -> str:
    log_path = os.path.join(
        project_root, 'data', 'results', 'pu_learning', 'pu_train.tsv'
    )
    if not os.path.exists(log_path):
        return '（尚无实验记录）'
    with open(log_path, 'r', encoding='utf-8') as f:
        return f.read()


def _build_user_prompt(dataset_report: str, algorithm_source: str) -> str:
    return (
        f'这是我的数据情况\n{dataset_report}\n\n'
        f'这是我的算法\n{algorithm_source}\n\n目标是挖掘出U样本中和P相似的样本，注意参数不要太精细避免过拟合'
        '帮我优化算法里使用到的参数，让算法效果达到最优。'
    )


def _build_system_prompt() -> str:
    keys_desc = ', '.join(PU_PARAM_KEYS)
    return f"""你是金融风控与机器学习专家，熟悉 PU Learning、LightGBM 与极度不平衡数据。

请根据数据集分析报告与 Bagging PU Learning 算法代码，推荐可使 AUC / Average Precision 更优的超参数。

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{PU_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，简要说明各参数选择依据>"
}}

params 中必须且仅包含以下键：{keys_desc}"""


def pu_params_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """比较两组 PU 超参（经 clamp 归一化后）是否完全一致。"""
    ca = _clamp_params(a)
    cb = _clamp_params(b)
    for key in PU_PARAM_KEYS:
        va, vb = ca[key], cb[key]
        if isinstance(va, (int, float)) or isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


def _clamp_params(params: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**DEFAULT_PU_PARAMS}
    for key in PU_PARAM_KEYS:
        if key not in params:
            continue
        val = params[key]
        if key == 'n_estimators':
            merged[key] = int(max(50, min(1000, float(val))))
        elif key == 'imbalance_ratio':
            merged[key] = float(max(0.1, min(1.0, float(val))))
        elif key == 'verbosity':
            merged[key] = int(max(-1, min(2, float(val))))
        elif key == 'learning_rate':
            merged[key] = float(max(0.001, min(0.5, float(val))))
        elif key == 'num_leaves':
            merged[key] = int(max(2, min(1000, float(val))))
        elif key == 'n_jobs':
            merged[key] = int(max(-1, min(64, float(val))))
        elif key == 'scale_pos_weight':
            merged[key] = float(max(0.1, min(100, float(val))))
        elif key == 'max_depth':
            merged[key] = int(max(1, min(50, float(val))))
        elif key == 'min_child_samples':
            merged[key] = int(max(1, min(1000, float(val))))
        elif key in ('subsample', 'colsample_bytree'):
            merged[key] = float(max(0.1, min(1.0, float(val))))
        elif key == 'num_boost_round':
            merged[key] = int(max(100, min(5000, float(val))))
    return merged


def optimize_pu_params_with_llm(
    project_root: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = 'gpt-4o-mini',
    log_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """
    调用大模型 API，结合 dataset_analysis_report.json 与 PU_bagging.py 推荐参数。
    完整回复会打印到 stderr 并写入 logs。
    """
    dataset_report = load_dataset_report(project_root)
    algorithm_source = _load_algorithm_source(project_root)

    client = create_openai_client(api_key, base_url, log_label='PU Param Optimizer')
    resolved_base = normalize_base_url(base_url) or os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    )

    user_prompt = _build_user_prompt(dataset_report, algorithm_source)
    print('[PU Param Optimizer] 正在调用大模型 API...', file=sys.stderr)
    print(f'[PU Param Optimizer] model={model}, base_url={resolved_base}', file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': _build_system_prompt()},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
    except (APIConnectionError, APITimeoutError) as e:
        raise ConnectionError(format_llm_connection_error(e)) from e
    except APIStatusError as e:
        raise RuntimeError(f'API 返回错误 {e.status_code}: {e.message}') from e
    raw_content = (response.choices[0].message.content or '').strip()
    _print_llm_response(raw_content, log_folder=log_folder)

    parsed = extract_json_object(raw_content)
    params_raw = parsed.get('params') or parsed
    if not isinstance(params_raw, dict):
        raise ValueError('大模型返回的 JSON 缺少 params 字段')

    recommended = _clamp_params(params_raw)
    reasoning = parsed.get('reasoning', '')

    return {
        'params': recommended,
        'reasoning': reasoning,
        'llm_response': raw_content,
        'model': model,
    }


def _build_autoresearch_system_prompt() -> str:
    keys_desc = ', '.join(PU_PARAM_KEYS)
    return f"""你是 autoresearch 实验助手。用户消息中会附带 puProgram.md（实验规程）、
dataset_analysis_report.json、PU_bagging.py 与 pu_train.tsv，请严格按规程推荐下一轮超参数以提升 F1。

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{PU_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，说明相对历史实验的调整思路，并引用 puProgram 中的相关约束若适用>"
}}

params 中必须且仅包含以下键：{keys_desc}
不可修改算法逻辑，仅调整上述超参数。"""


def _build_autoresearch_user_prompt(
    pu_program: str,
    dataset_report: str,
    algorithm_source: str,
    pu_train_log: str,
    iteration: int,
    best_f1: Optional[float],
    invalid_streak: int,
    param_retry_feedback: Optional[str] = None,
) -> str:
    best_str = f'{best_f1:.6g}' if best_f1 is not None else '尚无'
    parts = [
        f'## 当前轮次\niteration={iteration}, '
        f'historical_best_f1={best_str}, consecutive_non_improving_runs={invalid_streak}\n',
    ]
    if param_retry_feedback:
        parts.append(f'\n## 错误反馈\n{param_retry_feedback}\n')
    parts.extend([
        f'\n## puProgram.md（实验规程，必须遵守）\n{pu_program}\n',
        f'\n## dataset_analysis_report.json\n{dataset_report}\n',
        f'\n## PU_bagging.py\n{algorithm_source}\n',
        f'\n## pu_train.tsv（最新历史，请据此避免重复无效参数并改进 F1）\n{pu_train_log}\n',
        '\n请给出下一轮实验参数；params 必须与当前轮基准参数不同。',
    ])
    return ''.join(parts)


def _log_autoresearch_prompt_payload(
    project_root: str,
    *,
    pu_program: str,
    dataset_report: str,
    algorithm_source: str,
    pu_train_log: str,
    system_prompt: str,
    user_prompt: str,
) -> None:
    """记录本次 LLM 请求已附带的文件与体积，便于核对是否上传完整。"""
    file_specs = [
        ('puProgram.md', os.path.join(project_root, 'skill', 'puProgram.md'), pu_program),
        (
            'bagging.py',
            os.path.join(os.path.dirname(__file__), 'bagging.py'),
            algorithm_source,
        ),
        (
            'dataset_analysis_report.json',
            os.path.join(project_root, 'data', 'results', 'dataset_analysis_report.json'),
            dataset_report,
        ),
        (
            'pu_train.tsv',
            os.path.join(project_root, 'data', 'results', 'pu_learning', 'pu_train.tsv'),
            pu_train_log,
        ),
    ]
    print('[PU Autoresearch] LLM 请求上下文（均已写入 user/system messages）:', file=sys.stderr)
    for label, path, content in file_specs:
        preview = (content.strip().splitlines() or [''])[0][:72]
        print(
            f'  - {label}: exists={os.path.exists(path)}, payload={len(content)} chars, '
            f'first_line={preview!r}',
            file=sys.stderr,
        )
    print(
        f'  - messages: system={len(system_prompt)} chars, user={len(user_prompt)} chars',
        file=sys.stderr,
    )


def suggest_pu_params_autoresearch(
    project_root: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = 'gpt-4o-mini',
    log_folder: Optional[str] = None,
    iteration: int = 1,
    best_f1: Optional[float] = None,
    invalid_streak: int = 0,
    param_retry_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """
    autoresearch：结合 puProgram.md、数据集报告、算法源码与最新 pu_train.tsv 推荐参数。
    每次调用都会重新读取 pu_train.tsv。
    """
    dataset_report = load_dataset_report(project_root)
    algorithm_source = _load_algorithm_source(project_root)
    pu_program = _load_pu_program(project_root)
    pu_train_log = _load_pu_train_log(project_root)

    client = create_openai_client(api_key, base_url, log_label='PU Autoresearch')
    resolved_base = normalize_base_url(base_url) or os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    )

    system_prompt = _build_autoresearch_system_prompt()
    user_prompt = _build_autoresearch_user_prompt(
        pu_program,
        dataset_report,
        algorithm_source,
        pu_train_log,
        iteration,
        best_f1,
        invalid_streak,
        param_retry_feedback=param_retry_feedback,
    )

    retry_tag = ' (params 未更新重试)' if param_retry_feedback else ''
    print(
        f'[PU Autoresearch] iteration={iteration}{retry_tag}, '
        f'调用 LLM model={model}, base={resolved_base}',
        file=sys.stderr,
    )
    _log_autoresearch_prompt_payload(
        project_root,
        pu_program=pu_program,
        dataset_report=dataset_report,
        algorithm_source=algorithm_source,
        pu_train_log=pu_train_log,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=0.4,
            max_tokens=2500,
        )
    except (APIConnectionError, APITimeoutError) as e:
        raise ConnectionError(format_llm_connection_error(e)) from e
    except APIStatusError as e:
        raise RuntimeError(f'API 返回错误 {e.status_code}: {e.message}') from e

    raw_content = (response.choices[0].message.content or '').strip()
    _print_llm_response(raw_content, log_folder=log_folder)

    parsed = extract_json_object(raw_content)
    params_raw = parsed.get('params') or parsed
    if not isinstance(params_raw, dict):
        raise ValueError('大模型返回的 JSON 缺少 params 字段')

    recommended = _clamp_params(params_raw)
    reasoning = parsed.get('reasoning', '')

    return {
        'params': recommended,
        'reasoning': reasoning,
        'llm_response': raw_content,
        'model': model,
    }
