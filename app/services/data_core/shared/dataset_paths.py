"""训练/测试集路径解析（split / full 模式，供 CLI 与训练流水线共用）。"""
import json
import os
from typing import Optional, Tuple


def default_full_dataset_path(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'uploads', 'full_dataset.csv')


def _preferences_path(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results', 'dataset_preferences.json')


def _full_path_from_preferences(project_root: str) -> Optional[str]:
    path = _preferences_path(project_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            prefs = json.load(f)
        value = prefs.get('full_dataset_path')
        if value is not None and str(value).strip():
            return str(value).strip()
    except (json.JSONDecodeError, OSError):
        pass
    return None


def resolve_full_dataset_path(
    project_root: str,
    *,
    full_path: Optional[str] = None,
) -> str:
    """
    解析 full 模式下的全量 CSV 绝对路径。
    优先级：显式参数 > dataset_preferences.json > uploads 默认 > data/train.csv
    """
    if full_path is not None and str(full_path).strip():
        path = os.path.abspath(str(full_path).strip())
    else:
        pref_path = _full_path_from_preferences(project_root)
        path = os.path.abspath(pref_path) if pref_path else default_full_dataset_path(project_root)

    if os.path.isfile(path):
        return path

    fallback = os.path.join(project_root, 'data', 'train.csv')
    if os.path.isfile(fallback):
        return os.path.abspath(fallback)

    extra = f' 或 {fallback}' if path != os.path.abspath(fallback) else ''
    raise FileNotFoundError(f'未找到全量数据集: {path}{extra}')


def default_split_paths(project_root: str) -> Tuple[str, str]:
    upload_dir = os.path.join(project_root, 'data', 'uploads')
    return (
        os.path.join(upload_dir, 'train_dataset.csv'),
        os.path.join(upload_dir, 'test_dataset.csv'),
    )


def _path_from_session(session: dict, key: str) -> Optional[str]:
    value = session.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_split_paths(
    project_root: str,
    *,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    use_session: bool = True,
) -> Tuple[str, str]:
    """
    解析 split 模式下的 train/test CSV 绝对路径。
    优先级：显式参数 > pu_session.json > data/uploads 默认文件。
    """
    if use_session and (train_path is None or test_path is None):
        from app.services.data_core.shared.pu_session import read_pu_session

        session = read_pu_session(project_root) or {}
        if train_path is None:
            train_path = _path_from_session(session, 'train_path')
        if test_path is None:
            test_path = _path_from_session(session, 'test_path')

    default_train, default_test = default_split_paths(project_root)
    train = os.path.abspath(train_path or default_train)
    test = os.path.abspath(test_path or default_test)

    if not os.path.isfile(train):
        raise FileNotFoundError(f'未找到训练集: {train}')
    if not os.path.isfile(test):
        raise FileNotFoundError(f'未找到测试集: {test}')
    return train, test


def resolve_train_path(
    project_root: str,
    *,
    train_path: Optional[str] = None,
    use_session: bool = True,
) -> str:
    """解析仅需训练集的 split 场景（FE / MLBase compare 等）。"""
    train, _ = resolve_split_paths(
        project_root,
        train_path=train_path,
        test_path=None,
        use_session=use_session,
    )
    return train
