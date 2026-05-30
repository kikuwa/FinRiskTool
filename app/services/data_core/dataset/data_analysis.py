from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _normalize_binary_value(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return int(value)
        return None
    if isinstance(value, str):
        s = value.strip()
        if s in ('0', '1'):
            return int(s)
    return None


def _is_binary_01_column(series: pd.Series) -> bool:
    normalized = set()
    for value in series.dropna().unique():
        v = _normalize_binary_value(value)
        if v is None:
            return False
        normalized.add(v)
    return bool(normalized) and normalized.issubset({0, 1})


def detect_binary_01_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    candidates = []
    for col in df.columns:
        series = df[col]
        if not _is_binary_01_column(series):
            continue
        non_null = series.dropna()
        counts = non_null.map(_normalize_binary_value).value_counts().sort_index()
        candidates.append({
            'name': col,
            'unique_values': [int(k) for k in counts.index.tolist()],
            'distribution': {str(int(k)): int(v) for k, v in counts.items()},
        })
    return candidates


def _label_column_status(df: pd.DataFrame, label_col: str) -> Tuple[bool, str]:
    col = (label_col or '').strip()
    if not col:
        return False, 'empty_name'
    if col not in df.columns:
        return False, 'not_found'
    if df[col].isna().all():
        return False, 'all_null'
    return True, 'ok'


def _feature_missing_detail(df: pd.DataFrame, label_col: str) -> Dict[str, Dict]:
    n = len(df)
    if n == 0:
        return {}
    missing = df.isnull().sum()
    detail = {}
    for col in df.columns:
        cnt = int(missing[col])
        if cnt <= 0:
            continue
        ratio = cnt / n
        detail[col] = {
            'count': cnt,
            'ratio': round(ratio, 6),
            'ratio_pct': f'{ratio:.2%}',
        }
    return detail


def drop_features_by_missing_ratio(
    df: pd.DataFrame,
    max_missing_ratio: float,
    label_col: str = 'label',
) -> Tuple[pd.DataFrame, List[str]]:
    if not 0 <= max_missing_ratio <= 1:
        raise ValueError('缺失比例阈值须在 0~1 之间')

    n = len(df)
    if n == 0:
        return df.copy(), []

    to_drop = []
    for col in df.columns:
        if col == label_col:
            continue
        ratio = df[col].isnull().sum() / n
        if max_missing_ratio >= 1.0:
            if ratio >= 1.0:
                to_drop.append(col)
        elif ratio > max_missing_ratio:
            to_drop.append(col)

    if not to_drop:
        return df.copy(), []

    return df.drop(columns=to_drop), to_drop


def analyze_dataset(df, label_col='label', missing_threshold_applied: float = None):
    stats = {}

    stats['feature_count'] = df.shape[1]
    stats['sample_count'] = df.shape[0]

    label_col = (label_col or '').strip()
    stats['label_col_requested'] = label_col or None
    label_ok, label_reason = _label_column_status(df, label_col)

    if label_ok:
        stats['label_status'] = 'ok'
        value_counts = df[label_col].value_counts(dropna=False).to_dict()
        stats['label_distribution'] = {str(k): int(v) for k, v in value_counts.items()}
        total = df.shape[0]
        stats['label_ratio'] = {
            str(k): f'{v/total:.2%}' for k, v in value_counts.items()
        }
    else:
        stats['label_status'] = 'invalid'
        stats['label_invalid_reason'] = label_reason
        if label_reason == 'empty_name':
            stats['label_hint'] = '未填写标签列名，请在上方输入标签列或从下方候选列中选择。'
        elif label_reason == 'not_found':
            stats['label_hint'] = (
                f'标签列「{label_col}」不存在于当前数据集中，请检查列名或从下方 0/1 候选列中选择。'
            )
        else:
            stats['label_hint'] = (
                f'标签列「{label_col}」全部为空值，请更换列名或从下方 0/1 候选列中选择。'
            )
        stats['label_distribution'] = None
        stats['label_ratio'] = None
        stats['binary_01_feature_candidates'] = detect_binary_01_columns(df)
        stats['binary_01_feature_names'] = [
            c['name'] for c in stats['binary_01_feature_candidates']
        ]

    dtypes = df.dtypes.value_counts().to_dict()
    stats['dtypes'] = {str(k): v for k, v in dtypes.items()}

    missing_detail = _feature_missing_detail(df, label_col)
    stats['missing_features_count'] = len(missing_detail)
    stats['full_missing_stats'] = missing_detail

    if missing_detail:
        sorted_cols = sorted(
            missing_detail.items(),
            key=lambda x: x[1]['ratio'],
            reverse=True,
        )
        stats['top_missing_features'] = dict(sorted_cols[:5])
    else:
        stats['top_missing_features'] = {}

    stats['missing_threshold_options'] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if missing_threshold_applied is not None:
        stats['missing_threshold_applied'] = missing_threshold_applied
        stats['missing_threshold_applied_pct'] = f'{missing_threshold_applied:.0%}'

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if label_col in numeric_cols:
        numeric_cols = numeric_cols.drop(label_col)

    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers > 0:
            outlier_counts[col] = int(outliers)

    sorted_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
    stats['full_outlier_stats'] = {k: v for k, v in sorted_outliers}
    stats['top_outlier_features'] = {k: v for k, v in sorted_outliers[:5]}

    return stats
