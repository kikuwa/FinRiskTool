"""MLBase autoresearch 训练日志 ml_train.tsv。"""
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from app.services.data_core.mlbase.param_optimizer import (
    format_ml_train_params,
    merge_mlbase_params,
)

ML_TRAIN_LOG_FILENAME = 'ml_train.tsv'
ML_TRAIN_LOG_HEADER = [
    'variant',
    '参数',
    '验证阈值',
    '验证召回',
    '验证精度',
    '时间',
]


def ml_train_log_path(output_dir: str) -> str:
    return os.path.join(output_dir, ML_TRAIN_LOG_FILENAME)


def _format_log_value(value: Union[float, str, None]) -> str:
    if value is None:
        return 'NAN'
    if isinstance(value, str):
        return value
    if isinstance(value, float) and np.isnan(value):
        return 'NAN'
    return f'{float(value):.6g}'


def append_ml_train_log(
    output_dir: str,
    variant: str,
    threshold: Union[float, str, None],
    recall: Union[float, str, None],
    precision: Union[float, str, None],
    time_value: str,
    params: Union[dict, str, None] = None,
) -> str:
    """追加一条 MLBase 训练记录到 ml_train.tsv。"""
    os.makedirs(output_dir, exist_ok=True)
    log_path = ml_train_log_path(output_dir)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    if isinstance(params, dict):
        params_str = format_ml_train_params(params)
    elif params is None:
        params_str = 'NAN'
    else:
        params_str = str(params)

    row = [
        variant,
        params_str,
        _format_log_value(threshold),
        _format_log_value(recall),
        _format_log_value(precision),
        time_value,
    ]

    with open(log_path, 'a', encoding='utf-8', newline='') as f:
        if write_header:
            f.write('|'.join(ML_TRAIN_LOG_HEADER) + '\n')
        f.write('|'.join(row) + '\n')

    return log_path


def append_ml_train_log_timeout(
    output_dir: str,
    variant: str,
    params: Union[dict, None] = None,
) -> str:
    return append_ml_train_log(
        output_dir,
        variant,
        'NAN',
        'NAN',
        'NAN',
        'timeout',
        params=params,
    )


def _read_precision_values_from_ml_train(
    project_root: str,
    output_dir: Optional[str] = None,
) -> List[float]:
    output_dir = output_dir or os.path.join(
        project_root, 'data', 'results', 'feature_selection'
    )
    path = ml_train_log_path(output_dir)
    if not os.path.exists(path):
        return []

    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return []

    header = lines[0].split('|')
    if '验证精度' not in header:
        return []
    prec_idx = header.index('验证精度')

    values: List[float] = []
    for row_line in lines[1:]:
        row = row_line.split('|')
        if len(row) <= prec_idx:
            continue
        raw = row[prec_idx].strip()
        if raw.upper() == 'NAN' or raw.lower() == 'timeout':
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def read_best_precision_from_ml_train(
    project_root: str,
    output_dir: Optional[str] = None,
) -> Optional[float]:
    values = _read_precision_values_from_ml_train(project_root, output_dir)
    return max(values) if values else None


def read_latest_precision_from_ml_train(
    project_root: str,
    output_dir: Optional[str] = None,
) -> Optional[float]:
    values = _read_precision_values_from_ml_train(project_root, output_dir)
    return values[-1] if values else None


def load_ml_train_log(project_root: str) -> str:
    path = ml_train_log_path(
        os.path.join(project_root, 'data', 'results', 'feature_selection')
    )
    if not os.path.isfile(path):
        return '（尚无 ml_train.tsv 历史记录）'
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def _feature_selection_output_dir(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results', 'feature_selection')


def _read_last_ml_train_row(
    output_dir: str,
    variant: str,
) -> Optional[Dict[str, str]]:
    path = ml_train_log_path(output_dir)
    if not os.path.isfile(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return None

    header = lines[0].split('|')
    last_row = None
    for row_line in lines[1:]:
        row = row_line.split('|')
        if len(row) != len(header):
            continue
        record = dict(zip(header, row))
        if record.get('variant') == variant:
            last_row = record
    return last_row


def seed_ml_train_baseline_from_comparison(
    project_root: str,
    variant: str,
    *,
    initial_params: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """将 mlbase_comparison.json 中现有实验结果写入 ml_train.tsv 作为 baseline 参考。"""
    from app.services.data_core.mlbase.comparison import load_comparison_from_disk

    comparison = load_comparison_from_disk(project_root)
    if not comparison:
        return None

    experiment = comparison.get(variant)
    if not experiment or experiment.get('val_precision') is None:
        return None

    output_dir = _feature_selection_output_dir(project_root)
    merged_params = merge_mlbase_params(
        initial_params or {},
        experiment.get('params') or {},
    )
    params_str = format_ml_train_params(merged_params)
    precision_str = _format_log_value(experiment.get('val_precision'))

    last_row = _read_last_ml_train_row(output_dir, variant)
    if last_row:
        last_params = (last_row.get('参数') or '').strip()
        last_precision = (last_row.get('验证精度') or '').strip()
        if last_params == params_str and last_precision == precision_str:
            return None

    return append_ml_train_log(
        output_dir,
        variant,
        experiment.get('val_threshold'),
        experiment.get('val_recall'),
        experiment.get('val_precision'),
        'baseline',
        params=merged_params,
    )
