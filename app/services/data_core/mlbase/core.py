"""LightGBM 监督基线：单次训练 + 验证集阈值调优。"""
import json
import os
from typing import Any, Dict, List, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.shared.rus_sampling import (
    DEFAULT_RUS_RATIO,
    rus_dict_sampling_strategy,
)

POS_LABEL = 1
DEFAULT_RECALL_TARGET = 0.5
THRESHOLD_GRID_MIN = 0.01
THRESHOLD_GRID_MAX = 0.99
THRESHOLD_GRID_STEP = 0.01
DEFAULT_TRAIN_TEST_SPLIT = 0.25
DEFAULT_RANDOM_SEED = 42
MAX_TRAIN_ROWS = 80_000

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_N_ESTIMATORS = 150
DEFAULT_MAX_DEPTH = 6
DEFAULT_MIN_CHILD_SAMPLES = 50
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_REG_ALPHA = 0.1
DEFAULT_REG_LAMBDA = 0.1

MLBASE_COMPARISON_JSON = 'mlbase_comparison.json'
MLBASE_TEST_METRICS_JSON = 'mlbase_test_metrics.json'
MLBASE_TEST_PREDICTIONS_CSV = 'mlbase_test_predictions.csv'
MLBASE_VARIANTS = ('full_features', 'top_features')


def _scalar(value: Any, default: float, dtype=float) -> Union[float, int]:
    if value is None or value == '':
        return dtype(default)
    if isinstance(value, str):
        value = value.strip().split(',')[0].strip()
    return dtype(value)


def _tune_threshold(
    y_val, y_val_proba, recall_target: float
) -> Dict[str, float]:
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    n_steps = int(round((THRESHOLD_GRID_MAX - THRESHOLD_GRID_MIN) / THRESHOLD_GRID_STEP)) + 1
    for threshold in np.linspace(
        THRESHOLD_GRID_MIN, THRESHOLD_GRID_MAX, n_steps
    ):
        y_pred = (y_val_proba >= threshold).astype(int)
        val_recall = recall_score(y_val, y_pred, pos_label=POS_LABEL)
        val_precision = precision_score(y_val, y_pred, pos_label=POS_LABEL, zero_division=0)
        if val_recall >= recall_target and val_precision > best_precision:
            best_precision = val_precision
            best_recall = val_recall
            best_threshold = float(threshold)
    return {
        'threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall,
    }


def run_mlbase_experiment(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    label_col: str = 'label',
    recall_target: float = DEFAULT_RECALL_TARGET,
    reg_alpha: float = DEFAULT_REG_ALPHA,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    subsample: float = DEFAULT_SUBSAMPLE,
    random_state: int = DEFAULT_RANDOM_SEED,
    train_test_split_ratio: float = DEFAULT_TRAIN_TEST_SPLIT,
    rus_sampling_strategy: float = DEFAULT_RUS_RATIO,
    experiment_name: str = 'experiment',
) -> Dict[str, Any]:
    """单次 MLBase 实验：固定超参训练 + 验证集阈值调优。"""
    use_cols = [c for c in feature_cols if c in train_df.columns and c != label_col]
    if not use_cols:
        raise ValueError('无可用特征列')

    subset = train_df[[label_col] + use_cols].copy()
    loader = DataLoader(label_col=label_col, random_state=random_state)
    processed = loader.preprocess_data(subset, fit_encoders=True)

    data = processed.drop(columns=[label_col])
    y = processed[label_col]
    X_train, X_val, y_train, y_val = train_test_split(
        data,
        y,
        test_size=train_test_split_ratio,
        random_state=random_state,
        stratify=y,
    )

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    rus_strategy, rus_effective = rus_dict_sampling_strategy(
        y_train, rus_sampling_strategy
    )
    print(
        f'MLBase [{experiment_name}]: RUS 配置比={rus_sampling_strategy:.4f}，'
        f'实际少数/多数≈{rus_effective:.4f}'
    )
    rus = RandomUnderSampler(random_state=random_state, sampling_strategy=rus_strategy)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    if len(X_train_rus) > MAX_TRAIN_ROWS:
        X_train_rus, _, y_train_rus, _ = train_test_split(
            X_train_rus,
            y_train_rus,
            train_size=MAX_TRAIN_ROWS,
            random_state=random_state,
            stratify=y_train_rus,
        )
        print(
            f'MLBase [{experiment_name}]: 训练子样本 {len(X_train_rus)} 行'
            f'（上限 {MAX_TRAIN_ROWS}）'
        )

    params = {
        'learning_rate': float(learning_rate),
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_child_samples': int(min_child_samples),
        'subsample': float(subsample),
        'reg_alpha': float(reg_alpha),
        'reg_lambda': float(reg_lambda),
    }
    print(f'MLBase [{experiment_name}]: 单次训练 {params}，样本 {len(X_train_rus)} 行')

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='None',
        class_weight=weight_dict,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(X_train_rus, y_train_rus)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    threshold_metrics = _tune_threshold(y_val, y_val_proba, recall_target)

    return {
        'experiment_name': experiment_name,
        'feature_count': len(use_cols),
        'features': use_cols,
        'params': params,
        'class_weight': weight_dict,
        'recall_target': recall_target,
        'val_threshold': threshold_metrics['threshold'],
        'val_recall': threshold_metrics['recall'],
        'val_precision': threshold_metrics['precision'],
        'train_rows': len(X_train),
        'val_rows': len(X_val),
        'rus_effective_ratio': rus_effective,
    }


def _main_cli():
    """命令行入口（兼容旧 subprocess 调用）。"""
    from app.services.data_core.mlbase.comparison import run_mlbase_comparison

    project_root = os.getcwd()
    feature_list_env = os.environ.get('ML_FEATURE_LIST', '')
    common = dict(
        reg_alpha=float(os.environ.get('ML_REG_ALPHA', DEFAULT_REG_ALPHA)),
        reg_lambda=float(os.environ.get('ML_REG_LAMBDA', DEFAULT_REG_LAMBDA)),
        learning_rate=float(os.environ.get('ML_LEARNING_RATE', DEFAULT_LEARNING_RATE)),
        n_estimators=int(os.environ.get('ML_N_ESTIMATORS', DEFAULT_N_ESTIMATORS)),
        max_depth=int(os.environ.get('ML_MAX_DEPTH', DEFAULT_MAX_DEPTH)),
        min_child_samples=int(
            os.environ.get('ML_MIN_CHILD_SAMPLES', DEFAULT_MIN_CHILD_SAMPLES)
        ),
        subsample=float(os.environ.get('ML_SUBSAMPLE', DEFAULT_SUBSAMPLE)),
    )
    if feature_list_env.strip():
        features = [x.strip() for x in feature_list_env.split(',') if x.strip()]
        data_path = os.environ.get('ML_TRAIN_DATA_PATH', 'data/train.csv')
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_root, data_path)
        train_df = pd.read_csv(data_path)
        result = run_mlbase_experiment(train_df, features, **common)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = run_mlbase_comparison(project_root, **common)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    _main_cli()
