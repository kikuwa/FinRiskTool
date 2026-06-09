"""训练/测试集路径解析（split 模式，供 CLI 与训练流水线共用）。"""
import os
from typing import Optional, Tuple


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
