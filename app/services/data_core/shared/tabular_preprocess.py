"""表格数据统一预处理（缺失填充 + LabelEncoder）。"""
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class TabularEncoderState:
    label_col: str = 'label'
    fill_values: Dict[str, object] = field(default_factory=dict)
    label_encoders: Dict[str, LabelEncoder] = field(default_factory=dict)


def _resolve_numeric_fill(series: pd.Series, fallback: float = 0.0) -> float:
    """数值列填充值；全 NaN/Inf 列回退到 fallback。"""
    clean = pd.to_numeric(series, errors='coerce').replace(
        [np.inf, -np.inf], np.nan
    )
    for reducer in (clean.mean, clean.median):
        val = reducer()
        if pd.notna(val):
            return float(val)
    return float(fallback)


def preprocess_tabular_df(
    df: pd.DataFrame,
    label_col: str = 'label',
    encoder_state: Optional[TabularEncoderState] = None,
    fit: bool = True,
) -> tuple:
    """
    预处理特征列；label 列保持原样。

    Returns:
        (df_processed, encoder_state)
    """
    if encoder_state is None:
        encoder_state = TabularEncoderState(label_col=label_col)
    else:
        encoder_state.label_col = label_col

    df_processed = df.copy()

    for col in df_processed.columns:
        if col == label_col:
            continue

        if pd.api.types.is_numeric_dtype(df_processed[col]):
            col_series = pd.to_numeric(df_processed[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan
            )
            if fit:
                fill_val = _resolve_numeric_fill(col_series)
                encoder_state.fill_values[col] = fill_val
            else:
                stored = encoder_state.fill_values.get(col)
                if stored is None or (
                    isinstance(stored, (float, np.floating)) and pd.isna(stored)
                ):
                    fill_val = _resolve_numeric_fill(col_series)
                else:
                    fill_val = stored
            df_processed[col] = col_series.fillna(fill_val)
        else:
            mode_val = df_processed[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else 'Missing'
            if fit:
                encoder_state.fill_values[col] = fill_val
            else:
                fill_val = encoder_state.fill_values.get(col, fill_val)
            df_processed[col] = df_processed[col].fillna(fill_val)
            df_processed[col] = df_processed[col].astype(str)

            if fit:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                encoder_state.label_encoders[col] = le
            else:
                if col not in encoder_state.label_encoders:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col])
                    encoder_state.label_encoders[col] = le
                    continue
                le = encoder_state.label_encoders[col]
                known_classes = set(le.classes_)
                most_frequent = le.classes_[0]
                df_processed[col] = df_processed[col].apply(
                    lambda x: x if x in known_classes else most_frequent
                )
                df_processed[col] = le.transform(df_processed[col])

    return df_processed, encoder_state


def split_features_label(
    df: pd.DataFrame, label_col: str = 'label'
) -> tuple:
    """返回 (X, y)。"""
    if label_col not in df.columns:
        raise ValueError(f"标签列 '{label_col}' 不存在")
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)
    return X, y
