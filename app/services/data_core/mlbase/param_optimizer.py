"""MLBase 单次训练 LLM 参数推荐。"""
import os
import sys
from typing import Any, Dict, List, Optional

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.services.data_core.mlbase.core import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_CHILD_SAMPLES,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RECALL_TARGET,
    DEFAULT_REG_ALPHA,
    DEFAULT_REG_LAMBDA,
    DEFAULT_SUBSAMPLE,
)
from app.services.data_core.pu.param_optimizer import (
    _create_openai_client,
    _extract_json_object,
    _load_dataset_report,
    _normalize_base_url,
    format_llm_connection_error,
)

MLBASE_USER_PARAM_KEYS = ('recall_target',)

MLBASE_AI_PARAM_KEYS = (
    'learning_rate',
    'n_estimators',
    'max_depth',
    'min_child_samples',
    'subsample',
    'reg_alpha',
    'reg_lambda',
)

MLBASE_PARAM_KEYS = MLBASE_USER_PARAM_KEYS + MLBASE_AI_PARAM_KEYS

def format_ml_train_params(params: dict) -> str:
    """序列化超参，分号分隔，避免与 TSV 列分隔符冲突。"""
    if not params:
        return 'NAN'
    parts = [f'{key}={params[key]}' for key in MLBASE_PARAM_KEYS if key in params]
    return ';'.join(parts) if parts else 'NAN'


def extract_ml_train_params(kwargs: dict) -> Dict[str, Any]:
    return {key: kwargs[key] for key in MLBASE_PARAM_KEYS if key in kwargs}


DEFAULT_MLBASE_PARAMS: Dict[str, float] = {
    'recall_target': DEFAULT_RECALL_TARGET,
    'learning_rate': DEFAULT_LEARNING_RATE,
    'n_estimators': DEFAULT_N_ESTIMATORS,
    'max_depth': DEFAULT_MAX_DEPTH,
    'min_child_samples': DEFAULT_MIN_CHILD_SAMPLES,
    'subsample': DEFAULT_SUBSAMPLE,
    'reg_alpha': DEFAULT_REG_ALPHA,
    'reg_lambda': DEFAULT_REG_LAMBDA,
}


def _load_mlbase_source(project_root: str) -> str:
    path = os.path.join(os.path.dirname(__file__), 'core.py')
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_top_features_summary(project_root: str) -> str:
    path = os.path.join(
        project_root, 'data', 'results', 'feature_selection', 'top_features.csv'
    )
    if not os.path.isfile(path):
        raise FileNotFoundError('未找到 top_features.csv，请先在特征工程页确认特征')
    import pandas as pd
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    feat_col = 'feature_en' if 'feature_en' in cols else cols[0]
    names = df[feat_col].astype(str).tolist()
    preview = names[:30]
    suffix = f'\n... 共 {len(names)} 个特征' if len(names) > 30 else ''
    return f'列: {cols}\n特征数: {len(names)}\n特征列表(前30): {preview}{suffix}'


def _feature_selection_output_dir(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results', 'feature_selection')


def resolve_variant_feature_names(
    project_root: str,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
) -> List[str]:
    """返回当前 variant 下 MLBase 训练使用的完整特征名列表。"""
    from app.services.data_core.shared.data_loader import DataLoader

    output_dir = _feature_selection_output_dir(project_root)
    upload_dir = os.path.join(project_root, 'data', 'uploads')
    loader = DataLoader(label_col=label_col)
    if dataset_type == 'full':
        file_path = os.path.join(upload_dir, 'full_dataset.csv')
        if not os.path.isfile(file_path):
            file_path = os.path.join(project_root, 'data', 'train.csv')
        if not os.path.isfile(file_path):
            raise FileNotFoundError('未找到训练数据')
        train_df = loader._load_csv(file_path)
    else:
        train_path = os.path.join(upload_dir, 'train_dataset.csv')
        if not os.path.isfile(train_path):
            raise FileNotFoundError('未找到训练集')
        train_df = loader._load_csv(train_path)

    if variant == 'full_features':
        return [c for c in train_df.columns if c != label_col]

    top_path = os.path.join(output_dir, 'top_features.csv')
    if not os.path.isfile(top_path):
        raise FileNotFoundError('未找到 top_features.csv，请先在特征工程页确认特征')
    import pandas as pd
    top_df = pd.read_csv(top_path)
    feat_col = 'feature_en' if 'feature_en' in top_df.columns else top_df.columns[0]
    names = top_df[feat_col].astype(str).tolist()
    if not names:
        raise ValueError('top_features.csv 为空')
    return names


def detect_feature_set_changed(
    project_root: str,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
) -> bool:
    """当前特征集是否与 mlbase_comparison.json 中记录不一致。"""
    from app.services.data_core.mlbase.comparison import load_comparison_from_disk

    comparison = load_comparison_from_disk(project_root)
    if not comparison:
        return False
    experiment = comparison.get(variant)
    if not experiment:
        return False
    saved_features = experiment.get('features')
    if not saved_features:
        return False

    current = resolve_variant_feature_names(
        project_root, variant, label_col, dataset_type
    )
    return tuple(sorted(str(f) for f in current)) != tuple(
        sorted(str(f) for f in saved_features)
    )


def load_variant_features_for_llm(
    project_root: str,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
) -> str:
    """生成 autoresearch prompt 用的完整特征表文本。"""
    if variant == 'top_features':
        path = os.path.join(
            _feature_selection_output_dir(project_root), 'top_features.csv'
        )
        if not os.path.isfile(path):
            raise FileNotFoundError('未找到 top_features.csv，请先在特征工程页确认特征')
        with open(path, 'r', encoding='utf-8') as f:
            csv_content = f.read().strip()
        names = resolve_variant_feature_names(
            project_root, variant, label_col, dataset_type
        )
        return (
            f'variant=top_features\n特征数: {len(names)}\n'
            f'当前 MLBase 训练 strictly 仅使用 top_features.csv 中的下列特征，不可增删或更换 variant。\n\n'
            f'### top_features.csv（完整内容）\n{csv_content}'
        )

    names = resolve_variant_feature_names(
        project_root, variant, label_col, dataset_type
    )
    numbered = '\n'.join(f'{i + 1}. {name}' for i, name in enumerate(names))
    return (
        f'variant=full_features\n特征数: {len(names)}\n'
        f'当前 MLBase 训练 strictly 仅使用下列特征列（训练集除标签列外的全部列），'
        f'不可增删或更换 variant。\n\n'
        f'### 特征列表\n{numbered}'
    )


def _clamp_mlbase_params(params: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_MLBASE_PARAMS)
    for key in MLBASE_PARAM_KEYS:
        if key not in params:
            continue
        val = params[key]
        if key == 'recall_target':
            merged[key] = float(max(0.1, min(0.99, float(val))))
        elif key == 'learning_rate':
            merged[key] = float(max(0.001, min(0.3, float(val))))
        elif key == 'n_estimators':
            merged[key] = int(max(50, min(500, float(val))))
        elif key == 'max_depth':
            merged[key] = int(max(2, min(20, float(val))))
        elif key == 'min_child_samples':
            merged[key] = int(max(1, min(500, float(val))))
        elif key == 'subsample':
            merged[key] = float(max(0.1, min(1.0, float(val))))
        elif key in ('reg_alpha', 'reg_lambda'):
            merged[key] = float(max(0.0, min(10.0, float(val))))
    return merged


def _clamp_mlbase_ai_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """仅校验 AI 可调键（recall_target 由用户固定）。"""
    full = _clamp_mlbase_params(params)
    return {key: full[key] for key in MLBASE_AI_PARAM_KEYS}


def merge_mlbase_params(user_params: Dict[str, Any], ai_params: Dict[str, Any]) -> Dict[str, Any]:
    """合并用户设定与 AI 超参。"""
    merged = dict(DEFAULT_MLBASE_PARAMS)
    merged.update(_clamp_mlbase_params(user_params))
    merged.update(_clamp_mlbase_ai_params(ai_params))
    return merged


def _build_mlbase_system_prompt() -> str:
    keys_desc = ', '.join(MLBASE_AI_PARAM_KEYS)
    return f"""你是金融风控与机器学习专家，熟悉 LightGBM 监督学习与不平衡分类。

请根据数据集、MLBase 算法代码与已选 Top 特征列表，推荐单次训练的 LightGBM 超参数。
优化目标：在验证集 recall >= 用户设定的 recall_target 的前提下尽量提高 precision。
recall_target 由用户在界面固定，**禁止**在 params 中输出或修改。

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{MLBASE_AI_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，简要说明各参数选择依据>"
}}

params 中必须且仅包含以下键：{keys_desc}"""


def _build_mlbase_user_prompt(
    dataset_report: str,
    algorithm_source: str,
    top_features_summary: str,
) -> str:
    return (
        f'## dataset_analysis_report.json\n{dataset_report}\n\n'
        f'## MLBaseModel.py\n{algorithm_source}\n\n'
        f'## top_features.csv（已确认特征）\n{top_features_summary}\n\n'
        '请推荐 MLBase 单次训练参数。'
    )


def optimize_mlbase_params_with_llm(
    project_root: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = 'gpt-4o-mini',
) -> Dict[str, Any]:
    dataset_report = _load_dataset_report(project_root)
    algorithm_source = _load_mlbase_source(project_root)
    top_summary = _load_top_features_summary(project_root)

    client = _create_openai_client(api_key, base_url)
    resolved_base = _normalize_base_url(base_url) or os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    )
    print(
        f'[MLBase Param Optimizer] model={model}, base={resolved_base}',
        file=sys.stderr,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': _build_mlbase_system_prompt()},
                {
                    'role': 'user',
                    'content': _build_mlbase_user_prompt(
                        dataset_report, algorithm_source, top_summary
                    ),
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
        'params': _clamp_mlbase_ai_params(params_raw),
        'reasoning': parsed.get('reasoning', ''),
        'llm_response': raw_content,
        'model': model,
    }


def ml_params_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """比较两组 MLBase AI 超参（经 clamp 归一化后）是否完全一致。"""
    ca = _clamp_mlbase_ai_params(a)
    cb = _clamp_mlbase_ai_params(b)
    for key in MLBASE_AI_PARAM_KEYS:
        va, vb = ca[key], cb[key]
        if isinstance(va, (int, float)) or isinstance(vb, (int, float)):
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


def _load_ml_program(project_root: str) -> str:
    program_path = os.path.join(project_root, 'skill', 'MLProgram.md')
    if not os.path.isfile(program_path):
        raise FileNotFoundError('未找到 skill/MLProgram.md')
    with open(program_path, 'r', encoding='utf-8') as f:
        return f.read()


def _load_variant_features_summary(
    project_root: str,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
) -> str:
    return load_variant_features_for_llm(
        project_root, variant, label_col, dataset_type
    )


def _build_mlbase_autoresearch_system_prompt() -> str:
    keys_desc = ', '.join(MLBASE_AI_PARAM_KEYS)
    return f"""你是 MLBase autoresearch 实验助手。用户消息中会附带 MLProgram.md（实验规程）、
dataset_analysis_report.json、MLBase 算法代码与 ml_train.tsv，请严格按规程推荐下一轮超参数以提升验证集 precision。

recall_target 由用户在界面固定，**禁止**在 params 中输出或修改。

必须只输出一个 JSON 对象（不要用 markdown 代码块），结构如下：
{{
  "params": {{
    "{MLBASE_AI_PARAM_KEYS[0]}": <number>,
    ...
  }},
  "reasoning": "<中文，说明相对历史实验的调整思路，并引用 MLProgram 中的相关约束若适用>"
}}

params 中必须且仅包含以下键：{keys_desc}
不可修改算法逻辑、特征列表或 recall_target，仅调整上述 LightGBM 超参数。"""


def _build_mlbase_autoresearch_user_prompt(
    ml_program: str,
    dataset_report: str,
    algorithm_source: str,
    variant_summary: str,
    ml_train_log: str,
    variant: str,
    iteration: int,
    best_precision: Optional[float],
    invalid_streak: int,
    param_retry_feedback: Optional[str] = None,
    features_changed: bool = False,
) -> str:
    best_str = f'{best_precision:.6g}' if best_precision is not None else '尚无'
    parts = [
        f'## 当前轮次\niteration={iteration}, variant={variant}, '
        f'historical_best_val_precision={best_str}, '
        f'consecutive_non_improving_runs={invalid_streak}\n',
    ]
    if param_retry_feedback:
        parts.append(f'\n## 错误反馈\n{param_retry_feedback}\n')
    if features_changed:
        parts.append(
            '\n## 特征集已变更\n'
            '相对上次 ML 对比（mlbase_comparison.json）记录的特征集已变化。'
            '请勿假设 ml_train.tsv 中历史实验与当前特征完全一致；'
            '当前机器模型 strictly 仅使用下方特征表中的列进行训练。\n'
        )
    parts.extend([
        f'\n## MLProgram.md（实验规程，必须遵守）\n{ml_program}\n',
        f'\n## dataset_analysis_report.json\n{dataset_report}\n',
        f'\n## MLBase core.py\n{algorithm_source}\n',
        f'\n## 当前特征方案（模型仅使用下列特征）\n{variant_summary}\n',
        f'\n## ml_train.tsv（最新历史，请据此避免重复无效参数并改进验证精度）\n{ml_train_log}\n',
        '\n请给出下一轮实验参数；params 必须与当前轮基准参数不同。',
    ])
    return ''.join(parts)


def suggest_mlbase_params_autoresearch(
    project_root: str,
    api_key: str,
    *,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
    base_url: Optional[str] = None,
    model: str = 'gpt-4o-mini',
    log_folder: Optional[str] = None,
    iteration: int = 1,
    best_precision: Optional[float] = None,
    invalid_streak: int = 0,
    param_retry_feedback: Optional[str] = None,
) -> Dict[str, Any]:
    """autoresearch：结合 MLProgram.md、数据集报告、源码与 ml_train.tsv 推荐参数。"""
    from app.services.data_core.mlbase.train_log import load_ml_train_log
    from app.services.data_core.pu.param_optimizer import _print_llm_response

    dataset_report = _load_dataset_report(project_root)
    algorithm_source = _load_mlbase_source(project_root)
    ml_program = _load_ml_program(project_root)
    ml_train_log = load_ml_train_log(project_root)
    variant_summary = _load_variant_features_summary(
        project_root, variant, label_col, dataset_type
    )
    features_changed = detect_feature_set_changed(
        project_root, variant, label_col, dataset_type
    )

    client = _create_openai_client(api_key, base_url)
    resolved_base = _normalize_base_url(base_url) or os.environ.get(
        'OPENAI_BASE_URL', 'https://api.openai.com/v1'
    )

    system_prompt = _build_mlbase_autoresearch_system_prompt()
    user_prompt = _build_mlbase_autoresearch_user_prompt(
        ml_program,
        dataset_report,
        algorithm_source,
        variant_summary,
        ml_train_log,
        variant,
        iteration,
        best_precision,
        invalid_streak,
        param_retry_feedback=param_retry_feedback,
        features_changed=features_changed,
    )

    retry_tag = ' (params 未更新重试)' if param_retry_feedback else ''
    print(
        f'[MLBase Autoresearch] iteration={iteration}{retry_tag}, '
        f'variant={variant}, model={model}, base={resolved_base}',
        file=sys.stderr,
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

    parsed = _extract_json_object(raw_content)
    params_raw = parsed.get('params') or parsed
    if not isinstance(params_raw, dict):
        raise ValueError('大模型返回的 JSON 缺少 params 字段')

    return {
        'params': _clamp_mlbase_ai_params(params_raw),
        'reasoning': parsed.get('reasoning', ''),
        'llm_response': raw_content,
        'model': model,
    }
