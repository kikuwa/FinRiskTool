"""PU 会话配置持久化（供特征工程等模块读取先验）。"""
import json
import os
from typing import Any, Dict, Optional

PU_SESSION_FILENAME = 'pu_session.json'


def pu_session_path(project_root: str) -> str:
    return os.path.join(
        project_root, 'data', 'results', 'pu_learning', PU_SESSION_FILENAME
    )


def _label_col_from_preferences(project_root: str) -> Optional[str]:
    prefs_path = os.path.join(
        project_root, 'data', 'results', 'dataset_preferences.json'
    )
    if not os.path.isfile(prefs_path):
        return None
    try:
        with open(prefs_path, 'r', encoding='utf-8') as f:
            prefs = json.load(f)
        label_col = prefs.get('label_col')
        if label_col is not None and str(label_col).strip():
            return str(label_col).strip()
    except (json.JSONDecodeError, OSError):
        pass
    return None


def resolve_label_col(project_root: str, fallback: str = 'label') -> str:
    """数据集管理 preferences 优先，其次 pu_session，最后 fallback。"""
    lc = _label_col_from_preferences(project_root)
    if lc:
        return lc
    session = read_pu_session(project_root)
    if session:
        label_col = session.get('label_col')
        if label_col is not None and str(label_col).strip():
            return str(label_col).strip()
    return (fallback or 'label').strip()


def resolve_label_col_from_pu_session(
    project_root: str,
    fallback: str = 'label',
) -> str:
    return resolve_label_col(project_root, fallback)


def build_pu_session_payload(project_root: str, **fields: Any) -> Dict[str, Any]:
    """写入 pu_session 时统一标签列为 workflow 真源。"""
    payload = dict(fields)
    payload['label_col'] = resolve_label_col(
        project_root, payload.get('label_col') or 'label'
    )
    return payload


def write_pu_session(project_root: str, payload: Dict[str, Any]) -> str:
    path = pu_session_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def read_pu_session(project_root: str) -> Optional[Dict[str, Any]]:
    path = pu_session_path(project_root)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_dataset_type_from_pu_session(
    project_root: str,
    fallback: str = 'split',
) -> str:
    session = read_pu_session(project_root)
    if session:
        dt = session.get('dataset_type')
        if dt is not None and str(dt).strip():
            return str(dt).strip()
    return fallback
