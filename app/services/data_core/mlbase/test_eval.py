"""MLBase 在 test 集上的评估。"""
import json
import os
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from app.services.data_core.mlbase.core import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_CHILD_SAMPLES,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RECALL_TARGET,
    DEFAULT_REG_ALPHA,
    DEFAULT_REG_LAMBDA,
    DEFAULT_SUBSAMPLE,
    MAX_TRAIN_ROWS,
    MLBASE_COMPARISON_JSON,
    MLBASE_TEST_METRICS_JSON,
    MLBASE_TEST_PREDICTIONS_CSV,
    MLBASE_VARIANTS,
    POS_LABEL,
    _scalar,
)
from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.shared.rus_sampling import (
    DEFAULT_RUS_RATIO,
    rus_dict_sampling_strategy,
)


def _load_train_test_frames(
    project_root: str,
    label_col: str,
    dataset_type: str,
) -> tuple:
    upload_dir = os.path.join(project_root, 'data', 'uploads')
    loader = DataLoader(label_col=label_col)

    if dataset_type == 'full':
        raise ValueError('Test 集评估仅支持 split 模式（需 train_dataset.csv 与 test_dataset.csv）')

    train_path = os.path.join(upload_dir, 'train_dataset.csv')
    test_path = os.path.join(upload_dir, 'test_dataset.csv')
    if not os.path.isfile(train_path):
        raise FileNotFoundError('未找到训练集 train_dataset.csv')
    if not os.path.isfile(test_path):
        raise FileNotFoundError('未找到测试集 test_dataset.csv')

    train_df = loader._load_csv(train_path)
    test_df = loader._load_csv(test_path)
    loader.validate_data(train_df)
    loader.validate_data(test_df)
    if set(train_df.columns) != set(test_df.columns):
        raise ValueError('训练集与测试集列不一致')
    return train_df, test_df


def _fit_mlbase_on_full_train(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    *,
    label_col: str = 'label',
    random_state: int = DEFAULT_RANDOM_SEED,
    rus_sampling_strategy: float = DEFAULT_RUS_RATIO,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    subsample: float = DEFAULT_SUBSAMPLE,
    reg_alpha: float = DEFAULT_REG_ALPHA,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
) -> tuple:
    use_cols = [c for c in feature_cols if c in train_df.columns and c != label_col]
    if not use_cols:
        raise ValueError('无可用特征列')

    subset = train_df[[label_col] + use_cols].copy()
    loader = DataLoader(label_col=label_col, random_state=random_state)
    processed = loader.preprocess_data(subset, fit_encoders=True)

    data = processed.drop(columns=[label_col])
    y = processed[label_col]

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y
    )
    weight_dict = {0: float(class_weights[0]), 1: float(class_weights[1])}

    rus_strategy, _ = rus_dict_sampling_strategy(y, rus_sampling_strategy)
    rus = RandomUnderSampler(random_state=random_state, sampling_strategy=rus_strategy)
    X_rus, y_rus = rus.fit_resample(data, y)

    if len(X_rus) > MAX_TRAIN_ROWS:
        X_rus, _, y_rus, _ = train_test_split(
            X_rus,
            y_rus,
            train_size=MAX_TRAIN_ROWS,
            random_state=random_state,
            stratify=y_rus,
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

    model = lgb.LGBMClassifier(
        objective='binary',
        metric='None',
        class_weight=weight_dict,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )
    model.fit(X_rus, y_rus)
    return model, loader, use_cols, params, weight_dict


def run_mlbase_test_evaluation(
    project_root: str,
    *,
    variant: str = 'top_features',
    label_col: str = 'label',
    dataset_type: str = 'split',
    output_dir: Optional[str] = None,
    reg_alpha: float = DEFAULT_REG_ALPHA,
    reg_lambda: float = DEFAULT_REG_LAMBDA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
    subsample: float = DEFAULT_SUBSAMPLE,
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    if variant not in MLBASE_VARIANTS:
        raise ValueError(f'variant 须为 {MLBASE_VARIANTS}')

    output_dir = output_dir or os.path.join(
        project_root, 'data', 'results', 'feature_selection'
    )
    comparison_path = os.path.join(output_dir, MLBASE_COMPARISON_JSON)
    if not os.path.isfile(comparison_path):
        raise FileNotFoundError('请先完成 ML 对比（mlbase_comparison.json 不存在）')

    with open(comparison_path, 'r', encoding='utf-8') as f:
        comparison = json.load(f)

    exp = comparison.get(variant)
    if not exp:
        label = 'Top 特征' if variant == 'top_features' else '全量特征'
        raise ValueError(f'ML 对比结果中无 {label} 实验数据')

    feature_cols = list(exp.get('features') or [])
    if not feature_cols:
        raise ValueError(f'{variant} 无特征列表')

    threshold = float(exp.get('val_threshold', 0.5))
    recall_target = float(exp.get('recall_target', DEFAULT_RECALL_TARGET))

    train_df, test_df = _load_train_test_frames(project_root, label_col, dataset_type)

    model, loader, use_cols, params, class_weight = _fit_mlbase_on_full_train(
        train_df,
        feature_cols,
        label_col=label_col,
        random_state=random_state,
        learning_rate=_scalar(learning_rate, DEFAULT_LEARNING_RATE, float),
        n_estimators=_scalar(n_estimators, DEFAULT_N_ESTIMATORS, int),
        max_depth=_scalar(max_depth, DEFAULT_MAX_DEPTH, int),
        min_child_samples=_scalar(min_child_samples, DEFAULT_MIN_CHILD_SAMPLES, int),
        subsample=_scalar(subsample, DEFAULT_SUBSAMPLE, float),
        reg_alpha=_scalar(reg_alpha, DEFAULT_REG_ALPHA, float),
        reg_lambda=_scalar(reg_lambda, DEFAULT_REG_LAMBDA, float),
    )

    test_sub = test_df[[label_col] + use_cols].copy()
    test_processed = loader.preprocess_data(test_sub, fit_encoders=False)
    X_test = test_processed.drop(columns=[label_col])
    y_test = test_processed[label_col].astype(int)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    test_precision = precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    test_recall = recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)

    pred_df = pd.DataFrame({
        label_col: y_test.values,
        'prob': y_proba,
        'pred': y_pred,
    })
    pred_path = os.path.join(output_dir, MLBASE_TEST_PREDICTIONS_CSV)
    pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    metrics = {
        'variant': variant,
        'variant_label': 'Top 特征' if variant == 'top_features' else '全量特征',
        'feature_count': len(use_cols),
        'features': use_cols,
        'threshold': threshold,
        'recall_target': recall_target,
        'val_precision': exp.get('val_precision'),
        'val_recall': exp.get('val_recall'),
        'val_threshold': threshold,
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1': float(test_f1),
        'test_rows': int(len(y_test)),
        'train_rows': int(len(train_df)),
        'params': params,
        'class_weight': class_weight,
        'predictions_path': pred_path,
    }
    metrics_path = os.path.join(output_dir, MLBASE_TEST_METRICS_JSON)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f'MLBase Test [{variant}]: 阈值={threshold:.2f}, '
        f'精度={test_precision:.4f}, 召回={test_recall:.4f}, F1={test_f1:.4f}'
    )
    metrics['metrics_path'] = metrics_path
    return metrics


def load_mlbase_test_metrics(project_root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(
        project_root, 'data', 'results', 'feature_selection', MLBASE_TEST_METRICS_JSON
    )
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
