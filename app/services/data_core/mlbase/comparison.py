"""MLBase 全量 vs Top 特征对比。"""
import json
import os
from typing import Any, Dict, Optional

import pandas as pd

from app.services.data_core.mlbase.core import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_CHILD_SAMPLES,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RECALL_TARGET,
    DEFAULT_REG_ALPHA,
    DEFAULT_REG_LAMBDA,
    DEFAULT_SUBSAMPLE,
    _scalar,
    run_mlbase_experiment,
)
from app.services.data_core.shared.data_loader import DataLoader


def run_mlbase_comparison(
    project_root: str,
    *,
    label_col: str = 'label',
    dataset_type: str = 'split',
    train_path: Optional[str] = None,
    recall_target: float = DEFAULT_RECALL_TARGET,
    reg_alpha: float = DEFAULT_REG_ALPHA,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    subsample: float = DEFAULT_SUBSAMPLE,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """全量特征 vs top_features.csv 对比（各单次训练一次）。"""
    output_dir = output_dir or os.path.join(
        project_root, 'data', 'results', 'feature_selection'
    )
    upload_dir = os.path.join(project_root, 'data', 'uploads')
    loader = DataLoader(label_col=label_col)

    if dataset_type == 'full':
        file_path = os.path.join(upload_dir, 'full_dataset.csv')
        if not os.path.exists(file_path):
            file_path = os.path.join(project_root, 'data', 'train.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError('未找到训练数据')
        train_df = loader._load_csv(file_path)
    else:
        from app.services.data_core.shared.dataset_paths import resolve_train_path

        resolved = resolve_train_path(project_root, train_path=train_path)
        train_df = loader._load_csv(resolved)

    loader.validate_data(train_df)
    all_features = [c for c in train_df.columns if c != label_col]

    top_path = os.path.join(output_dir, 'top_features.csv')
    top_features = []
    warning = None
    if os.path.exists(top_path):
        top_df = pd.read_csv(top_path)
        top_features = top_df['feature_en'].astype(str).tolist()
    else:
        warning = '未找到 top_features.csv，Top 特征实验将跳过'

    kwargs = dict(
        label_col=label_col,
        recall_target=recall_target,
        reg_alpha=_scalar(reg_alpha, DEFAULT_REG_ALPHA, float),
        reg_lambda=_scalar(reg_lambda, DEFAULT_REG_LAMBDA, float),
        learning_rate=_scalar(learning_rate, DEFAULT_LEARNING_RATE, float),
        n_estimators=_scalar(n_estimators, DEFAULT_N_ESTIMATORS, int),
        max_depth=_scalar(max_depth, DEFAULT_MAX_DEPTH, int),
        min_child_samples=_scalar(min_child_samples, DEFAULT_MIN_CHILD_SAMPLES, int),
        subsample=_scalar(subsample, DEFAULT_SUBSAMPLE, float),
    )

    print(
        f'MLBase 对比: 数据 {len(train_df)} 行，全量特征 {len(all_features)} 个，单次训练'
    )

    full_result = run_mlbase_experiment(
        train_df,
        all_features,
        experiment_name='full_features',
        **kwargs,
    )

    comparison = {
        'full_features': full_result,
        'top_features': None,
        'warning': warning,
        'summary': {},
    }

    if top_features:
        top_result = run_mlbase_experiment(
            train_df,
            top_features,
            experiment_name='top_features',
            **kwargs,
        )
        comparison['top_features'] = top_result
        comparison['summary'] = {
            'full_val_precision': full_result['val_precision'],
            'top_val_precision': top_result['val_precision'],
            'full_val_recall': full_result['val_recall'],
            'top_val_recall': top_result['val_recall'],
            'precision_delta': top_result['val_precision'] - full_result['val_precision'],
            'recall_delta': top_result['val_recall'] - full_result['val_recall'],
            'full_feature_count': full_result['feature_count'],
            'top_feature_count': top_result['feature_count'],
        }

    out_path = os.path.join(output_dir, 'mlbase_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    comparison['output_path'] = out_path
    return comparison


def load_comparison_from_disk(project_root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(
        project_root, 'data', 'results', 'feature_selection', 'mlbase_comparison.json'
    )
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
