"""RFECV 递归特征消除，用于 ML 对比全量优于 Top 时重选特征。"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.feature_selection.pipeline import (
    FEATURE_CANDIDATES_CSV,
    TOP_FEATURES_CSV,
)

RFECV_META_JSON = 'rfecv_meta.json'


def _load_train_df(project_root: str, dataset_type: str, label_col: str) -> pd.DataFrame:
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
        train_path = os.path.join(upload_dir, 'train_dataset.csv')
        if not os.path.exists(train_path):
            raise FileNotFoundError('未找到训练集，请先完成数据分割')
        train_df = loader._load_csv(train_path)
    loader.validate_data(train_df)
    return train_df


def run_rfecv_reselection(
    project_root: str,
    *,
    label_col: str = 'label',
    dataset_type: str = 'split',
    output_dir: Optional[str] = None,
    cv_folds: int = 3,
    min_features_to_select: int = 5,
    log_fn=None,
) -> Dict[str, Any]:
    """
    对 feature_candidates 中全部特征运行 RFECV，更新候选表勾选状态。
    不写入 top_features.csv，需用户再次确认。
    """
    output_dir = output_dir or os.path.join(
        project_root, 'data', 'results', 'feature_selection'
    )
    cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    if not os.path.isfile(cand_path):
        raise FileNotFoundError('请先运行特征筛选')

    def _log(msg: str, level: str = 'info') -> None:
        if log_fn:
            log_fn(msg, level)
        else:
            print(f'[rfecv] {msg}')

    cand_df = pd.read_csv(cand_path)
    feature_cols = cand_df['feature_en'].astype(str).tolist()
    if not feature_cols:
        raise ValueError('候选特征列表为空')

    _log(f'加载训练数据 (dataset_type={dataset_type})…')
    train_df = _load_train_df(project_root, dataset_type, label_col)
    use_cols = [c for c in feature_cols if c in train_df.columns]
    if len(use_cols) < min_features_to_select:
        raise ValueError(f'可用特征不足 {min_features_to_select} 个')

    subset = train_df[[label_col] + use_cols].copy()
    loader = DataLoader(label_col=label_col)
    processed = loader.preprocess_data(subset, fit_encoders=True)
    X = processed.drop(columns=[label_col])
    y = processed[label_col].astype(int)

    n_features = X.shape[1]
    step = max(1, min(10, n_features // 10))
    _log(f'RFECV: {n_features} 特征, step={step}, cv={cv_folds}')

    estimator = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        verbosity=-1,
        n_jobs=-1,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring='f1',
        min_features_to_select=min_features_to_select,
        n_jobs=1,
    )
    selector.fit(X, y)

    support = selector.support_
    ranking = selector.ranking_
    selected_names = [use_cols[i] for i, s in enumerate(support) if s]
    _log(f'RFECV 完成: 选中 {len(selected_names)} / {n_features} 个特征')

    # 与 RFECV 相同配置的 LGB，在全特征上拟合一次，得到可展示的 per-feature 分值
    score_clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        verbosity=-1,
        n_jobs=-1,
        random_state=42,
    )
    score_clf.fit(X, y)
    feature_importances = score_clf.feature_importances_

    col_to_idx = {name: i for i, name in enumerate(use_cols)}
    rfecv_ranks: List[int] = []
    rfecv_selected: List[bool] = []
    rfecv_scores: List[Any] = []
    for feat in cand_df['feature_en'].astype(str):
        if feat not in col_to_idx:
            rfecv_ranks.append(9999)
            rfecv_selected.append(False)
            rfecv_scores.append('')
            continue
        idx = col_to_idx[feat]
        rfecv_ranks.append(int(ranking[idx]))
        rfecv_selected.append(bool(support[idx]))
        rfecv_scores.append(round(float(feature_importances[idx]), 6))

    cand_df['rfecv_rank'] = rfecv_ranks
    cand_df['rfecv_selected'] = rfecv_selected
    cand_df['rfecv_score'] = rfecv_scores
    cand_df['selected'] = rfecv_selected

    order = np.argsort(ranking)
    rank_map = {use_cols[i]: r + 1 for r, i in enumerate(order)}
    cand_df['rank'] = cand_df['feature_en'].astype(str).map(
        lambda f: rank_map.get(f, 9999)
    )

    top_path = os.path.join(output_dir, TOP_FEATURES_CSV)
    if os.path.isfile(top_path):
        os.unlink(top_path)
        _log('已清除旧的 top_features.csv，请重新确认特征', 'info')

    cand_df = cand_df.fillna('')
    cand_df.to_csv(cand_path, index=False, encoding='utf-8-sig')

    meta = {
        'n_features_in': n_features,
        'n_features_selected': int(selector.n_features_),
        'selected_features': selected_names,
        'step': step,
        'cv_folds': cv_folds,
        'scoring': 'f1',
    }
    if hasattr(selector, 'cv_results_'):
        scores = selector.cv_results_.get('mean_test_score')
        if scores is not None:
            meta['cv_mean_test_score'] = [
                float(x) for x in np.asarray(scores).tolist()
            ]

    meta_path = os.path.join(output_dir, RFECV_META_JSON)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        'output_dir': output_dir,
        'n_features_selected': meta['n_features_selected'],
        'selected_features': selected_names,
        'meta_path': meta_path,
        'meta': meta,
    }
