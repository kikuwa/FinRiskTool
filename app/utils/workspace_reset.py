"""服务启动时清空上一轮 data/results 与 data/uploads。"""
import os
import stat
import time
from typing import List, Tuple


def _chmod_writable(path: str) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _safe_unlink(path: str, retries: int = 3) -> bool:
    if not os.path.isfile(path):
        return True
    for attempt in range(retries):
        try:
            _chmod_writable(path)
            os.unlink(path)
            return True
        except OSError as exc:
            if attempt < retries - 1:
                time.sleep(0.05)
            else:
                print(f'[workspace] warning: failed to delete file {path}: {exc}')
    return False


def _clear_directory_contents(dir_path: str) -> Tuple[int, int]:
    """
    自底向上删除目录内所有文件与子目录（不删除 dir_path 本身）。
    返回 (deleted_files, failed_files)。
    """
    if not os.path.isdir(dir_path):
        return 0, 0

    deleted = 0
    failed = 0
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            fp = os.path.join(root, name)
            if _safe_unlink(fp):
                deleted += 1
            else:
                failed += 1
        for name in dirs:
            dp = os.path.join(root, name)
            try:
                os.rmdir(dp)
            except OSError as exc:
                print(f'[workspace] warning: failed to remove dir {dp}: {exc}')
    return deleted, failed


def _safe_rmtree(path: str) -> Tuple[int, int]:
    """删除目录树；Windows 上优先逐文件删除，避免整树 rmtree 因单文件占用失败。"""
    if not os.path.isdir(path):
        return 0, 0
    deleted, failed = _clear_directory_contents(path)
    try:
        os.rmdir(path)
    except OSError as exc:
        if os.path.isdir(path) and os.listdir(path):
            print(f'[workspace] warning: directory not empty after cleanup {path}: {exc}')
            failed += 1
        elif os.path.isdir(path):
            print(f'[workspace] warning: failed to remove {path}: {exc}')
    return deleted, failed


def clear_data_workspace(project_root: str) -> List[str]:
    """
    清空 data/results 全部内容及 data/uploads 下 CSV。
    返回已清理的顶层路径说明（用于日志）。
    """
    cleared: List[str] = []
    total_failed = 0
    results_dir = os.path.join(project_root, 'data', 'results')
    uploads_dir = os.path.join(project_root, 'data', 'uploads')

    if os.path.isdir(results_dir):
        for name in os.listdir(results_dir):
            target = os.path.join(results_dir, name)
            if os.path.isdir(target):
                _, failed = _safe_rmtree(target)
                total_failed += failed
            else:
                if not _safe_unlink(target):
                    total_failed += 1
        cleared.append('data/results/*')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'pu_learning'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'feature_selection'), exist_ok=True)

    if os.path.isdir(uploads_dir):
        for name in os.listdir(uploads_dir):
            if name.lower().endswith('.csv'):
                if not _safe_unlink(os.path.join(uploads_dir, name)):
                    total_failed += 1
        cleared.append('data/uploads/*.csv')
    os.makedirs(uploads_dir, exist_ok=True)

    if total_failed:
        cleared.append(f'warnings:{total_failed}_files_locked')
        print(
            f'[workspace] {total_failed} 个文件因被占用未能删除；'
            f'请关闭 Excel/其他程序占用 data/results 后重启'
        )

    return cleared


def should_clear_on_app_start() -> bool:
    """
    Debug 重载：仅在子进程清空（与 session_logging 一致），避免父进程与子进程各清一次，
    且父进程可能在子进程仍持有句柄时删除失败。
    """
    if os.environ.get('FINRISK_CLI') == '1':
        return False
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        return True
    if os.environ.get('FLASK_RUN_FROM_CLI') == 'true':
        return False
    return True
