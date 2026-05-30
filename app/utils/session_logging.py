"""服务启动时创建统一会话日志，并 tee stdout/stderr。"""
import multiprocessing as mp
import os
import sys
from datetime import datetime
from typing import Optional, TextIO, Union

ENV_SESSION_LOG_FILE = 'FINRISK_SESSION_LOG_FILE'


class _TeeStream:
    def __init__(self, original: TextIO, log_file: TextIO):
        self._original = original
        self._log_file = log_file

    @property
    def encoding(self) -> str:
        return getattr(self._original, 'encoding', None) or 'utf-8'

    def _normalize(self, data: Union[str, bytes]) -> str:
        if isinstance(data, bytes):
            return data.decode(self.encoding, errors='replace')
        return data

    def write(self, data: Union[str, bytes]) -> int:
        if not data:
            return 0
        text = self._normalize(data)
        self._original.write(text)
        self._original.flush()
        self._log_file.write(text)
        self._log_file.flush()
        return len(text)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._original, 'isatty', lambda: False)()

    def fileno(self):
        return self._original.fileno()


_session_log_file: Optional[TextIO] = None
_session_log_path: Optional[str] = None
_tee_attached = False


def _is_flask_reloader_parent() -> bool:
    """Flask debug 模式下 reloader 父进程不初始化日志（子进程会初始化）。"""
    return (
        os.environ.get('WERKZEUG_RUN_MAIN') != 'true'
        and os.environ.get('FLASK_RUN_FROM_CLI') == 'true'
    )


def _should_create_session_log() -> bool:
    if os.environ.get(ENV_SESSION_LOG_FILE):
        return False
    if mp.parent_process() is not None:
        return False
    if mp.current_process().name != 'MainProcess':
        return False
    if _is_flask_reloader_parent():
        return False
    return True


def _attach_tee(log_path: str, *, write_banner: bool = False) -> None:
    global _session_log_file, _session_log_path, _tee_attached

    if _tee_attached and _session_log_path == log_path:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    _session_log_path = log_path
    _session_log_file = open(log_path, 'a', encoding='utf-8')

    sys.stdout = _TeeStream(sys.__stdout__, _session_log_file)
    sys.stderr = _TeeStream(sys.__stderr__, _session_log_file)
    _tee_attached = True

    if write_banner:
        print(
            f"\n{'=' * 80}\n"
            f"Session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Log file: {log_path}\n"
            f"{'=' * 80}\n",
            end='',
        )


def setup_session_logging(log_folder: str) -> str:
    """
    每次 python run.py 启动仅创建一个 server_*.log；
    同一会话内（含 PU 训练子进程）全部追加写入同一文件。
    """
    global _session_log_path

    existing = os.environ.get(ENV_SESSION_LOG_FILE)
    if existing:
        _attach_tee(existing, write_banner=False)
        return existing

    if not _should_create_session_log():
        return ''

    os.makedirs(log_folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_folder, f'server_{timestamp}.log')
    os.environ[ENV_SESSION_LOG_FILE] = log_path
    _attach_tee(log_path, write_banner=True)
    return log_path


def attach_worker_session_logging() -> None:
    """PU 训练子进程：追加写入主进程会话日志，不新建文件。"""
    log_path = os.environ.get(ENV_SESSION_LOG_FILE)
    if not log_path:
        return
    marker = (
        f"\n--- PU worker ({mp.current_process().name}) "
        f"at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
    )
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(marker)
    _attach_tee(log_path, write_banner=False)


def get_session_log_path() -> Optional[str]:
    return _session_log_path or os.environ.get(ENV_SESSION_LOG_FILE)
