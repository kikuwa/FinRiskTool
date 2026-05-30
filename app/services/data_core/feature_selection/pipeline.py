"""特征选择：PN 构表 + MI 双阶段剔除 + LGB 稳定性实验。"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif

from app.services.data_core.shared.data_loader import DataLoader
from app.services.data_core.shared.rus_sampling import (
    DEFAULT_RUS_MIN_EFFECTIVE,
    DEFAULT_RUS_RATIO,
    rus_dict_sampling_strategy,
)
from app.services.data_core.shared.tabular_preprocess import (
    TabularEncoderState,
    preprocess_tabular_df,
    split_features_label,
)

FEATURE_CANDIDATES_CSV = 'feature_candidates.csv'
FEATURE_META_JSON = 'feature_selection_meta.json'
TOP_FEATURES_CSV = 'top_features.csv'
PN_SNAPSHOT_CSV = 'pn_train_snapshot.csv'
MI_FEATURE_THRESHOLD = 100
STABILITY_TOP_K = 50
STABILITY_MIN_HITS = 2
STABILITY_SEEDS = (42, 43, 44)
# 三次 LGB 前 RUS：重采样后 少数类数/多数类数（正样本/负样本，PN 表上）
# 0.07 表示约 1:14；PN 后天然约 1:10（~0.098），低于此比无法欠采样
STABILITY_RUS_RATIO = 0.15
STABILITY_RUS_MIN_EFFECTIVE = 0.12


@dataclass
class FeatureSelectionParams:
    estimated_positive_rate: float = 0.1
    mi_feature_threshold: int = MI_FEATURE_THRESHOLD
    stability_top_k: int = STABILITY_TOP_K
    stability_min_hits: int = STABILITY_MIN_HITS
    stability_rus_ratio: float = STABILITY_RUS_RATIO
    stability_lgb_n_estimators: int = 200
    stability_lgb_learning_rate: float = 0.05
    stability_lgb_max_depth: int = 6
    stability_lgb_num_leaves: int = 31


def resolve_feature_selection_params(
    fe_params: Optional[Dict[str, Any]] = None,
    *,
    estimated_positive_rate: Optional[float] = None,
) -> FeatureSelectionParams:
    from app.services.data_core.feature_selection.fe_optimizer import (
        DEFAULT_FE_PARAMS,
        _clamp_fe_params,
    )

    merged = _clamp_fe_params(dict(DEFAULT_FE_PARAMS))
    if fe_params:
        merged.update(_clamp_fe_params(dict(fe_params)))
    if estimated_positive_rate is not None:
        merged['estimated_positive_rate'] = _clamp_fe_params({
            'estimated_positive_rate': estimated_positive_rate,
        })['estimated_positive_rate']
    return FeatureSelectionParams(
        estimated_positive_rate=float(merged['estimated_positive_rate']),
        mi_feature_threshold=int(merged['mi_feature_threshold']),
        stability_top_k=int(merged['stability_top_k']),
        stability_min_hits=int(merged['stability_min_hits']),
        stability_rus_ratio=float(merged['stability_rus_ratio']),
        stability_lgb_n_estimators=int(merged['stability_lgb_n_estimators']),
        stability_lgb_learning_rate=float(merged['stability_lgb_learning_rate']),
        stability_lgb_max_depth=int(merged['stability_lgb_max_depth']),
        stability_lgb_num_leaves=int(merged['stability_lgb_num_leaves']),
    )


def load_feature_mapping(feature_file: str) -> Dict[str, str]:
    if not os.path.exists(feature_file):
        return {}
    feature_map = {}
    try:
        with open(feature_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 2)
                if len(parts) < 3:
                    continue
                feature_map[parts[0]] = parts[1]
    except Exception as exc:
        print(f'加载特征映射失败: {exc}')
    return feature_map


def build_pn_train_for_feature_selection(
    train_df: pd.DataFrame,
    train_predictions_path: str,
    estimated_positive_rate: float,
    label_col: str = 'label',
) -> Tuple[pd.DataFrame, int]:
    """
    按 PU 先验删除 prob 最高的 U（原 label=0），剩余 0=N、1=P。
    返回 (pn_train, removed_count)。
    """
    if not os.path.exists(train_predictions_path):
        raise FileNotFoundError(
            f'未找到 {train_predictions_path}，请先在 PU Learning 页完成训练'
        )
    if not 0 < estimated_positive_rate <= 1:
        raise ValueError('预估正样本比例须在 (0, 1] 之间')

    preds = pd.read_csv(train_predictions_path)
    if len(preds) != len(train_df):
        raise ValueError(
            f'train_predictions 行数 ({len(preds)}) 与训练集 ({len(train_df)}) 不一致'
        )

    prob_col = 'prob' if 'prob' in preds.columns else None
    if prob_col is None:
        for c in preds.columns:
            if c != label_col and pd.api.types.is_numeric_dtype(preds[c]):
                prob_col = c
                break
    if prob_col is None:
        raise ValueError('train_predictions 中未找到 prob 列')

    work = train_df.copy()
    work['_pu_prob'] = preds[prob_col].values

    u_mask = work[label_col] == 0
    u_df = work.loc[u_mask].sort_values('_pu_prob', ascending=False)
    n_remove = max(1, int(np.ceil(len(u_df) * estimated_positive_rate)))
    remove_idx = u_df.index[:n_remove]

    pn_train = work.drop(index=remove_idx).drop(columns=['_pu_prob'])
    return pn_train.reset_index(drop=True), len(remove_idx)


def _preprocess_xy(
    df: pd.DataFrame,
    label_col: str,
    encoder_state: Optional[TabularEncoderState] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, TabularEncoderState]:
    processed, state = preprocess_tabular_df(
        df, label_col=label_col, encoder_state=encoder_state, fit=fit
    )
    X, y = split_features_label(processed, label_col)
    return X, y, state


def _mi_bottom_half_features(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> Set[str]:
    """返回 MI 排名后 50% 的特征名集合。"""
    scores = mutual_info_classif(X, y, random_state=random_state)
    names = X.columns.tolist()
    order = np.argsort(scores)[::-1]
    n = len(names)
    bottom_start = n // 2
    bottom_indices = order[bottom_start:]
    return {names[i] for i in bottom_indices}


def _fit_encoder_state_for_features(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
) -> TabularEncoderState:
    """在 PN（或训练）子集上拟合编码器，供后续 LGB 稳定性使用。"""
    sub = df[[label_col] + feature_cols].copy()
    for col in feature_cols:
        if col == label_col:
            continue
        if not pd.api.types.is_numeric_dtype(sub[col]):
            sub[col] = sub[col].astype(str)
    _, _, state = _preprocess_xy(sub, label_col, fit=True)
    return state


def filter_features_by_dual_mi(
    raw_train: pd.DataFrame,
    pn_train: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    mi_feature_threshold: int = MI_FEATURE_THRESHOLD,
) -> Tuple[List[str], TabularEncoderState]:
    """在 raw 与 pn 上均落入 MI 后 50% 的特征剔除（两次都不重要才删）。"""
    if len(feature_cols) <= mi_feature_threshold:
        state = _fit_encoder_state_for_features(pn_train, label_col, feature_cols)
        return list(feature_cols), state

    raw_sub = raw_train[[label_col] + feature_cols].copy()
    pn_sub = pn_train[[label_col] + feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(raw_sub[col]):
            raw_sub[col] = raw_sub[col].astype(str)
        if not pd.api.types.is_numeric_dtype(pn_sub[col]):
            pn_sub[col] = pn_sub[col].astype(str)

    X_raw, y_raw, state = _preprocess_xy(raw_sub, label_col, fit=True)
    bottom_raw = _mi_bottom_half_features(X_raw, y_raw)

    X_pn, y_pn, state = _preprocess_xy(pn_sub, label_col, encoder_state=state, fit=False)
    bottom_pn = _mi_bottom_half_features(X_pn, y_pn)

    drop = bottom_raw & bottom_pn
    kept = [c for c in feature_cols if c not in drop]
    print(
        f'MI 双阶段: 总特征 {len(feature_cols)}, '
        f'剔除 {len(drop)}, 保留 {len(kept)}'
    )
    state = _fit_encoder_state_for_features(pn_train, label_col, kept)
    return kept, state


def _lgb_stability_selection(
    pn_train: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    encoder_state: TabularEncoderState,
    top_k: int = STABILITY_TOP_K,
    min_hits: int = STABILITY_MIN_HITS,
    rus_ratio: float = STABILITY_RUS_RATIO,
    lgb_n_estimators: int = 200,
    lgb_learning_rate: float = 0.05,
    lgb_max_depth: int = 6,
    lgb_num_leaves: int = 31,
) -> Tuple[Dict[str, int], Dict[str, float], float]:
    """
    三次欠采样 + LGB，统计进入 TopK 的次数与平均 importance。
  返回 (hit_counts, avg_imp, effective_rus_ratio)。
    """
    pn_sub = pn_train[[label_col] + feature_cols].copy()
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(pn_sub[col]):
            pn_sub[col] = pn_sub[col].astype(str)
    X, y, _ = _preprocess_xy(pn_sub, label_col, encoder_state=encoder_state, fit=False)

    hit_counts: Dict[str, int] = {f: 0 for f in feature_cols}
    importance_sum: Dict[str, float] = {f: 0.0 for f in feature_cols}

    rus_strategy, effective_ratio = rus_dict_sampling_strategy(
        y, rus_ratio, min_effective_ratio=STABILITY_RUS_MIN_EFFECTIVE
    )
    ordered = sorted(
        zip(*np.unique(y, return_counts=True)), key=lambda x: x[1]
    )
    minor_label, n_minor = int(ordered[0][0]), int(ordered[0][1])
    major_label, n_major = int(ordered[1][0]), int(ordered[1][1])
    n_major_target = rus_strategy[major_label]
    n_after = n_minor + n_major_target
    print(
        f'LGB 稳定性: RUS 配置比={rus_ratio:.4f}，实际少数/多数≈{effective_ratio:.4f}；'
        f'{major_label} {n_major}->{n_major_target}，{minor_label}={n_minor}，'
        f'每次训练约 {n_after} 行 × {len(STABILITY_SEEDS)} 次'
    )

    for seed in STABILITY_SEEDS:
        rus = RandomUnderSampler(
            random_state=seed, sampling_strategy=rus_strategy
        )
        X_rus, y_rus = rus.fit_resample(X, y)
        clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=int(lgb_n_estimators),
            learning_rate=float(lgb_learning_rate),
            num_leaves=int(lgb_num_leaves),
            max_depth=int(lgb_max_depth),
            verbosity=-1,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_rus, y_rus)
        imp = clf.feature_importances_
        names = X.columns.tolist()
        order = np.argsort(imp)[::-1][:top_k]
        for idx in order:
            feat = names[idx]
            hit_counts[feat] += 1
            importance_sum[feat] += float(imp[idx])

    stable = {
        f: hit_counts[f]
        for f in feature_cols
        if hit_counts[f] >= min_hits
    }
    avg_imp = {
        f: importance_sum[f] / max(hit_counts[f], 1)
        for f in stable
    }
    return hit_counts, avg_imp, effective_ratio


def run_feature_selection_pipeline(
    train_df: pd.DataFrame,
    label_col: str = 'label',
    output_dir: str = 'data/results/feature_selection',
    estimated_positive_rate: float = 0.1,
    train_predictions_path: Optional[str] = None,
    project_root: Optional[str] = None,
    top_k: Optional[int] = None,
    fe_params: Optional[Dict[str, Any]] = None,
) -> Dict:
    """完整特征选择流水线。"""
    os.makedirs(output_dir, exist_ok=True)
    project_root = project_root or os.getcwd()
    params = resolve_feature_selection_params(
        fe_params, estimated_positive_rate=estimated_positive_rate
    )
    if top_k is not None:
        params.stability_top_k = int(top_k)
    feature_map = load_feature_mapping(
        os.path.join(project_root, 'config', '全部特征.txt')
    )

    feature_cols = [c for c in train_df.columns if c != label_col]
    raw_train = train_df.copy()

    if not train_predictions_path:
        train_predictions_path = os.path.join(
            project_root, 'data', 'results', 'pu_learning', 'train_predictions.csv'
        )

    pn_train, removed_u = build_pn_train_for_feature_selection(
        raw_train,
        train_predictions_path,
        params.estimated_positive_rate,
        label_col=label_col,
    )
    pn_train.to_csv(
        os.path.join(output_dir, PN_SNAPSHOT_CSV), index=False, encoding='utf-8-sig'
    )

    kept_cols, encoder_state = filter_features_by_dual_mi(
        raw_train,
        pn_train,
        label_col,
        feature_cols,
        mi_feature_threshold=params.mi_feature_threshold,
    )

    hit_counts: Dict[str, int] = {f: 0 for f in kept_cols}
    avg_imp: Dict[str, float] = {f: 0.0 for f in kept_cols}
    skip_stability = len(kept_cols) <= params.stability_top_k
    effective_rus_ratio = None

    if not skip_stability:
        hit_counts, avg_imp, effective_rus_ratio = _lgb_stability_selection(
            pn_train,
            kept_cols,
            label_col,
            encoder_state,
            top_k=params.stability_top_k,
            min_hits=params.stability_min_hits,
            rus_ratio=params.stability_rus_ratio,
            lgb_n_estimators=params.stability_lgb_n_estimators,
            lgb_learning_rate=params.stability_lgb_learning_rate,
            lgb_max_depth=params.stability_lgb_max_depth,
            lgb_num_leaves=params.stability_lgb_num_leaves,
        )

    candidates = []
    for feat in kept_cols:
        hits = hit_counts.get(feat, 0)
        if skip_stability:
            selected = True
            hits = 3
        else:
            selected = hits >= params.stability_min_hits
        candidates.append({
            'feature_en': feat,
            'feature_zh': feature_map.get(feat, feat),
            'mi_kept': True,
            'stable_hits': hits,
            'lgb_importance_mean': round(avg_imp.get(feat, 0.0), 6),
            'selected': selected,
        })

    if not skip_stability:
        selected_rows = [r for r in candidates if r['selected']]
        selected_rows.sort(key=lambda r: r['lgb_importance_mean'], reverse=True)
        if len(selected_rows) > params.stability_top_k:
            top_set = {r['feature_en'] for r in selected_rows[: params.stability_top_k]}
            for row in candidates:
                if row['feature_en'] not in top_set and row['selected']:
                    row['selected'] = False

    candidates.sort(key=lambda r: r['lgb_importance_mean'], reverse=True)
    for i, row in enumerate(candidates):
        row['rank'] = i + 1

    cand_df = pd.DataFrame(candidates)
    if 'rank' in cand_df.columns:
        cand_df['rank'] = cand_df['rank'].apply(
            lambda v: '' if v == '' or (isinstance(v, float) and pd.isna(v)) else v
        )
    cand_df = cand_df.fillna('')
    cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    cand_df.to_csv(cand_path, index=False, encoding='utf-8-sig')

    meta = {
        'label_col': label_col,
        'fe_params': asdict(params),
        'pn_removed_u_count': removed_u,
        'raw_row_count': len(raw_train),
        'pn_row_count': len(pn_train),
        'initial_feature_count': len(feature_cols),
        'after_mi_feature_count': len(kept_cols),
        'stability_skipped': skip_stability,
        'stability_rus_ratio_effective': effective_rus_ratio,
        'default_selected_count': int(cand_df['selected'].sum()),
    }
    meta_path = os.path.join(output_dir, FEATURE_META_JSON)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    default_selected = cand_df[cand_df['selected']]['feature_en'].tolist()
    print(f'特征选择完成: 候选 {len(candidates)}，默认选中 {len(default_selected)}')

    return {
        'output_dir': output_dir,
        'feature_candidates_path': cand_path,
        'feature_meta_path': meta_path,
        'candidates': candidates,
        'default_selected': default_selected,
        'meta': meta,
    }


def confirm_top_features(
    output_dir: str,
    selected_features: List[str],
    candidates_path: Optional[str] = None,
) -> str:
    """业务确认后写入 top_features.csv。"""
    candidates_path = candidates_path or os.path.join(
        output_dir, FEATURE_CANDIDATES_CSV
    )
    if not os.path.exists(candidates_path):
        raise FileNotFoundError('请先运行特征筛选')

    cand_df = pd.read_csv(candidates_path)
    valid = set(cand_df['feature_en'].astype(str))
    unknown = [f for f in selected_features if f not in valid]
    if unknown:
        raise ValueError(f'以下特征不在候选列表中: {unknown[:5]}')

    imp_col = 'lgb_importance_mean'
    if imp_col not in cand_df.columns:
        imp_col = None

    rows = []
    for feat in selected_features:
        match = cand_df[cand_df['feature_en'].astype(str) == str(feat)]
        if match.empty:
            continue
        row = match.iloc[0]
        imp = float(row[imp_col]) if imp_col and pd.notna(row.get(imp_col)) else 0.0
        rows.append({
            'feature_en': feat,
            'feature_zh': row.get('feature_zh', feat),
            'lgb_importance_mean': imp,
            'selected_by': 'user',
        })

    if not rows:
        raise ValueError('无有效特征可写入')

    out_df = pd.DataFrame(rows)
    if imp_col:
        out_df = out_df.sort_values('lgb_importance_mean', ascending=False)

    out_path = os.path.join(output_dir, TOP_FEATURES_CSV)
    out_df.drop(columns=['lgb_importance_mean'], errors='ignore').to_csv(
        out_path, index=False, encoding='utf-8-sig'
    )
    return out_path


def read_top_features(output_dir: str) -> List[str]:
    path = os.path.join(output_dir, TOP_FEATURES_CSV)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df['feature_en'].astype(str).tolist()
