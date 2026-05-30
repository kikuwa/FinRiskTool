import os
from typing import Dict, Optional, Tuple

import chardet
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.services.data_core.shared.tabular_preprocess import (
    TabularEncoderState,
    preprocess_tabular_df,
)


class DataLoader:
    def __init__(self, label_col: str = 'label', test_size: float = 0.3, random_state: int = 42):
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.encoder_state = TabularEncoderState(label_col=label_col)
        # 兼容旧属性访问
        self.label_encoders = self.encoder_state.label_encoders
        self.fill_values = self.encoder_state.fill_values

    def load_full_dataset(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载完整数据集并自动划分训练集和测试集"""
        df = self._load_csv(file_path)
        self.validate_data(df)

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.label_col],
        )
        return train_df, test_df

    def load_train_test_split(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分别加载训练集和测试集"""
        train_df = self._load_csv(train_path)
        test_df = self._load_csv(test_path)

        self.validate_data(train_df)
        self.validate_data(test_df)

        if set(train_df.columns) != set(test_df.columns):
            raise ValueError('训练集和测试集的列不一致')

        return train_df, test_df

    def _detect_encoding(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(100000))
        return result['encoding']

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'文件未找到: {file_path}')

        if not file_path.lower().endswith('.csv'):
            raise ValueError('只允许上传 CSV 文件')

        detected_encoding = self._detect_encoding(file_path)
        encodings_to_try = []
        if detected_encoding:
            encodings_to_try.append(detected_encoding)

        common_encodings = [
            'utf-8',
            'gbk',
            'gb18030',
            'big5',
            'latin-1',
            'utf-16',
            'cp1252',
        ]

        for enc in common_encodings:
            if enc not in encodings_to_try:
                encodings_to_try.append(enc)

        last_error = None
        for encoding in encodings_to_try:
            try:
                return pd.read_csv(file_path, encoding=encoding, low_memory=False)
            except (UnicodeDecodeError, LookupError) as e:
                last_error = e
                continue

        try:
            return pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
        except Exception as e:
            raise ValueError(
                f'无法读取文件。尝试了以下编码: {encodings_to_try}。错误: {last_error}'
            ) from e

    def validate_data(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError('数据集为空，请上传有效数据')

        if self.label_col not in df.columns:
            raise ValueError(f"指定的标签列 '{self.label_col}' 不存在于数据集中")

        unique_labels = df[self.label_col].unique()
        if len(unique_labels) != 2:
            raise ValueError(
                f"标签列 '{self.label_col}' 必须是二分类（只能有 2 个唯一值），"
                f'当前有 {len(unique_labels)} 个: {unique_labels}'
            )

        if len(df.columns) < 2:
            raise ValueError('数据集必须至少包含一个特征列（除标签列外）')

    def preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        self.encoder_state.label_col = self.label_col
        df_processed, self.encoder_state = preprocess_tabular_df(
            df,
            label_col=self.label_col,
            encoder_state=self.encoder_state,
            fit=fit_encoders,
        )
        self.label_encoders = self.encoder_state.label_encoders
        self.fill_values = self.encoder_state.fill_values
        return df_processed
