import os
import time

import chardet
import pandas as pd
from sklearn.model_selection import train_test_split


def _detect_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']


def _load_csv_with_encoding(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'文件未找到: {file_path}')

    detected_encoding = _detect_encoding(file_path)
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


def safe_to_csv(df: pd.DataFrame, path: str, max_retries: int = 5, retry_delay: float = 0.4) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f'.{os.path.basename(path)}.tmp')

    last_error = None
    for _ in range(max_retries):
        try:
            df.to_csv(temp_path, index=False, encoding='utf-8')
            os.replace(temp_path, path)
            return
        except PermissionError as e:
            last_error = e
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            time.sleep(retry_delay)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    raise PermissionError(
        f"无法写入 '{path}'，文件可能被 Excel 或其他程序占用。请关闭后重试。"
    ) from last_error


def split_data(input_file, train_output, test_output, test_size=0.3, label_col='label'):
    print(f'读取数据文件: {input_file}')
    df = _load_csv_with_encoding(input_file)

    if label_col not in df.columns:
        raise KeyError(f"标签列 '{label_col}' 不存在于数据集中。可用列: {list(df.columns)}")

    print(f'原始数据形状: {df.shape}')
    print('原始标签分布:')
    print(df[label_col].value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=42,
    )

    print(f'\n训练集形状: {train_df.shape}')
    print('训练集标签分布:')
    print(train_df[label_col].value_counts())

    print(f'\n测试集形状: {test_df.shape}')
    print('测试集标签分布:')
    print(test_df[label_col].value_counts())

    safe_to_csv(train_df, train_output)
    safe_to_csv(test_df, test_output)

    print('\n数据分割完成！')
    print(f'训练集已保存到: {train_output}')
    print(f'测试集已保存到: {test_output}')


if __name__ == '__main__':
    split_data(
        input_file='data/generated_data.csv',
        train_output='data/train.csv',
        test_output='data/test.csv',
        test_size=0.3,
    )
