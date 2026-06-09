"""MLBase 单次 variant 训练（供 HTTP 与 autoresearch 共用）。"""
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from app.services.data_core.mlbase.core import run_mlbase_experiment
from app.services.data_core.mlbase.param_optimizer import extract_ml_train_params
from app.services.data_core.mlbase.train_log import (
    append_ml_train_log,
    append_ml_train_log_timeout,
)
from app.services.data_core.shared.data_loader import DataLoader

ML_RUN_TIMEOUT_SECONDS = 1200  # 20 分钟默认
ML_STOP_POLL_INTERVAL_SECONDS = 0.5

_active_ml_proc: Optional[mp.Process] = None
_active_ml_stop_event: Any = None
_active_ml_pid: Optional[int] = None
_active_ml_proc_lock = threading.Lock()
_ml_user_stop_pending = False


class MLBaseTrainTimeoutError(TimeoutError):
    """MLBase 训练超过允许时长。"""


class MLBaseTrainStoppedError(InterruptedError):
    """MLBase 训练被用户主动停止。"""


def mark_ml_user_stop_requested() -> None:
    """标记用户已请求停止（子进程尚未启动时 terminate 也会生效）。"""
    global _ml_user_stop_pending
    _ml_user_stop_pending = True


def _user_stop_requested(stop_checker: Optional[Callable[[], bool]] = None) -> bool:
    if _ml_user_stop_pending:
        return True
    return bool(stop_checker and stop_checker())


def _check_stop_event(stop_event: Any) -> None:
    if stop_event is not None and stop_event.is_set():
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')


def _kill_active_mlbase_process() -> bool:
    """强制终止已注册的 MLBase 训练子进程（与超时相同 taskkill 路径）。"""
    with _active_ml_proc_lock:
        proc = _active_ml_proc
        pid = _active_ml_pid
        stop_event = _active_ml_stop_event

    if stop_event is not None:
        stop_event.set()

    killed = False
    if proc is not None:
        try:
            if proc.is_alive():
                print('[MLBase] 收到停止信号，正在终止训练子进程…')
                _terminate_process_tree(proc)
                print('[MLBase] 训练子进程已终止')
                killed = True
        except (ValueError, OSError):
            pass

    if not killed and pid is not None and sys.platform == 'win32':
        print(f'[MLBase] 按 PID {pid} 终止训练进程树…')
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            check=False,
        )
        killed = True

    return killed


def _start_mlbase_stop_watchdog(
    stop_checker: Optional[Callable[[], bool]],
    stop_event: Any,
) -> tuple:
    cancel = threading.Event()

    def _watch() -> None:
        while not cancel.is_set():
            if _user_stop_requested(stop_checker):
                print('[MLBase] 看门狗：检测到用户停止请求，终止训练…')
                _kill_active_mlbase_process()
                return
            cancel.wait(ML_STOP_POLL_INTERVAL_SECONDS)

    thread = threading.Thread(
        target=_watch,
        name='mlbase-train-stop-watchdog',
        daemon=True,
    )
    thread.start()
    return cancel, thread


def terminate_active_mlbase_training() -> bool:
    """终止正在运行的 MLBase 训练子进程（autoresearch 停止时调用）。"""
    mark_ml_user_stop_requested()
    return _kill_active_mlbase_process() or _ml_user_stop_pending


def _terminate_process_tree(proc: mp.Process) -> None:
    pid = proc.pid
    if pid is None:
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return
    if sys.platform == 'win32':
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            capture_output=True,
            check=False,
        )
    else:
        proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _load_train_df(
    project_root: str,
    label_col: str,
    dataset_type: str,
    *,
    train_path: Optional[str] = None,
) -> pd.DataFrame:
    from app.services.data_core.shared.dataset_paths import resolve_train_path

    upload_dir = os.path.join(project_root, 'data', 'uploads')
    loader = DataLoader(label_col=label_col)
    if dataset_type == 'full':
        file_path = os.path.join(upload_dir, 'full_dataset.csv')
        if not os.path.exists(file_path):
            file_path = os.path.join(project_root, 'data', 'train.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError('未找到训练数据')
        return loader._load_csv(file_path)
    resolved = resolve_train_path(project_root, train_path=train_path)
    return loader._load_csv(resolved)


def _resolve_feature_cols(
    project_root: str,
    train_df: pd.DataFrame,
    label_col: str,
    variant: str,
    output_dir: str,
) -> List[str]:
    if variant == 'full_features':
        return [c for c in train_df.columns if c != label_col]
    top_path = os.path.join(output_dir, 'top_features.csv')
    if not os.path.isfile(top_path):
        raise FileNotFoundError('未找到 top_features.csv，请先在特征工程页确认特征')
    top_df = pd.read_csv(top_path)
    top_features = top_df['feature_en'].astype(str).tolist()
    if not top_features:
        raise ValueError('top_features.csv 为空')
    return top_features


def load_mlbase_training_data(
    project_root: str,
    *,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
    train_path: Optional[str] = None,
    stop_event: Any = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """在训练子进程内加载数据并解析特征列（便于停止时 taskkill 整个子进程）。"""
    _check_stop_event(stop_event)
    output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
    train_df = _load_train_df(
        project_root, label_col, dataset_type, train_path=train_path,
    )
    _check_stop_event(stop_event)
    loader = DataLoader(label_col=label_col)
    loader.validate_data(train_df)
    _check_stop_event(stop_event)
    feature_cols = _resolve_feature_cols(
        project_root, train_df, label_col, variant, output_dir
    )
    return train_df, feature_cols


def _mlbase_worker(kwargs: dict, result_queue: mp.Queue) -> None:
    stop_event = kwargs.pop('stop_event', None)
    try:
        if stop_event and stop_event.is_set():
            result_queue.put({'status': 'stopped'})
            return

        if 'train_df' not in kwargs or 'feature_cols' not in kwargs:
            project_root = kwargs.pop('project_root')
            variant = kwargs.pop('variant')
            label_col = kwargs.get('label_col', 'label')
            dataset_type = kwargs.pop('dataset_type', 'split')
            train_path = kwargs.pop('train_path', None)
            train_df, feature_cols = load_mlbase_training_data(
                project_root,
                variant=variant,
                label_col=label_col,
                dataset_type=dataset_type,
                train_path=train_path,
                stop_event=stop_event,
            )
            kwargs['train_df'] = train_df
            kwargs['feature_cols'] = feature_cols
            kwargs['experiment_name'] = variant

        t0 = time.monotonic()
        result = run_mlbase_experiment(**kwargs)
        elapsed = time.monotonic() - t0
        if stop_event and stop_event.is_set():
            result_queue.put({'status': 'stopped'})
            return
        result_queue.put({
            'status': 'ok',
            'result': result,
            'elapsed': elapsed,
        })
    except MLBaseTrainStoppedError:
        result_queue.put({'status': 'stopped'})
    except Exception as exc:
        result_queue.put({
            'status': 'error',
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })


def run_mlbase_experiment_with_timeout(
    *,
    timeout_seconds: int = ML_RUN_TIMEOUT_SECONDS,
    stop_checker: Optional[Callable[[], bool]] = None,
    output_dir: str,
    variant: str,
    log_params: dict,
    **experiment_kwargs,
) -> Dict[str, Any]:
    """子进程运行 run_mlbase_experiment，支持超时与用户停止。"""
    global _active_ml_proc, _active_ml_stop_event, _active_ml_pid, _ml_user_stop_pending

    os.makedirs(output_dir, exist_ok=True)

    if _user_stop_requested(stop_checker):
        _ml_user_stop_pending = False
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')

    ctx = mp.get_context('spawn')
    stop_event = ctx.Event()
    result_queue = ctx.Queue()
    worker_kwargs = dict(experiment_kwargs)
    worker_kwargs['variant'] = variant
    worker_kwargs['stop_event'] = stop_event

    proc = ctx.Process(
        target=_mlbase_worker,
        args=(worker_kwargs, result_queue),
        daemon=True,
    )

    with _active_ml_proc_lock:
        _active_ml_proc = proc
        _active_ml_stop_event = stop_event
        _active_ml_pid = None

    proc.start()
    with _active_ml_proc_lock:
        _active_ml_pid = proc.pid

    watchdog_cancel, watchdog_thread = _start_mlbase_stop_watchdog(stop_checker, stop_event)

    if _user_stop_requested(stop_checker):
        watchdog_cancel.set()
        watchdog_thread.join(timeout=2)
        _kill_active_mlbase_process()
        with _active_ml_proc_lock:
            _active_ml_proc = None
            _active_ml_stop_event = None
            _active_ml_pid = None
        _ml_user_stop_pending = False
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')

    deadline = time.monotonic() + timeout_seconds
    stopped_by_user = False

    try:
        while proc.is_alive():
            if _user_stop_requested(stop_checker):
                stopped_by_user = True
                _kill_active_mlbase_process()
                break
            if time.monotonic() >= deadline:
                break
            proc.join(timeout=ML_STOP_POLL_INTERVAL_SECONDS)
    finally:
        watchdog_cancel.set()
        watchdog_thread.join(timeout=2)
        if stopped_by_user or _user_stop_requested(stop_checker):
            _kill_active_mlbase_process()
        with _active_ml_proc_lock:
            _active_ml_proc = None
            _active_ml_stop_event = None
            _active_ml_pid = None
        if stopped_by_user or _user_stop_requested(stop_checker):
            _ml_user_stop_pending = False

    if stopped_by_user or _user_stop_requested(stop_checker):
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')

    if proc.is_alive():
        stop_event.set()
        _terminate_process_tree(proc)
        _ml_user_stop_pending = False
        append_ml_train_log_timeout(output_dir, variant, log_params)
        raise MLBaseTrainTimeoutError(
            f'MLBase 运行超过 {timeout_seconds // 60} 分钟已终止'
        )

    message = None
    while message is None:
        if _user_stop_requested(stop_checker):
            _kill_active_mlbase_process()
            raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')
        try:
            message = result_queue.get(timeout=ML_STOP_POLL_INTERVAL_SECONDS)
        except queue.Empty:
            if result_queue.empty():
                if _user_stop_requested(stop_checker):
                    raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')
                append_ml_train_log_timeout(output_dir, variant, log_params)
                raise MLBaseTrainTimeoutError('MLBase 未返回结果（可能超时）')
            continue

    _ml_user_stop_pending = False
    if message['status'] == 'stopped':
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')
    if message['status'] == 'error':
        raise RuntimeError(message.get('error', 'MLBase 训练失败'))

    if _user_stop_requested(stop_checker):
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')

    result = message['result']
    elapsed = message['elapsed']
    append_ml_train_log(
        output_dir,
        variant,
        result['val_threshold'],
        result['val_recall'],
        result['val_precision'],
        f'{elapsed:.2f}',
        params=log_params,
    )
    result['run_time_seconds'] = elapsed
    return result


def execute_mlbase_variant_training(
    project_root: str,
    *,
    variant: str,
    label_col: str = 'label',
    dataset_type: str = 'split',
    train_path: Optional[str] = None,
    ml_params: Dict[str, Any] = None,
    timeout_seconds: int = None,
    stop_checker: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """加载数据并在指定 variant 下运行 MLBase 实验。"""
    if variant not in ('full_features', 'top_features'):
        raise ValueError('variant 须为 full_features 或 top_features')

    if stop_checker and stop_checker():
        raise MLBaseTrainStoppedError('用户已停止 MLBase 训练')

    ml_params = dict(ml_params or {})
    timeout_seconds = timeout_seconds or ML_RUN_TIMEOUT_SECONDS
    output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')

    log_params = extract_ml_train_params(ml_params)
    experiment_kwargs = dict(
        project_root=project_root,
        label_col=label_col,
        dataset_type=dataset_type,
        train_path=train_path,
        recall_target=float(ml_params.get('recall_target', 0.5)),
        reg_alpha=float(ml_params.get('reg_alpha', 0.1)),
        reg_lambda=float(ml_params.get('reg_lambda', 0.1)),
        learning_rate=float(ml_params.get('learning_rate', 0.05)),
        n_estimators=int(ml_params.get('n_estimators', 150)),
        max_depth=int(ml_params.get('max_depth', 6)),
        min_child_samples=int(ml_params.get('min_child_samples', 50)),
        subsample=float(ml_params.get('subsample', 0.8)),
    )

    result = run_mlbase_experiment_with_timeout(
        timeout_seconds=timeout_seconds,
        stop_checker=stop_checker,
        output_dir=output_dir,
        variant=variant,
        log_params=log_params,
        **experiment_kwargs,
    )
    return {
        'success': True,
        'variant': variant,
        'experiment': result,
        'val_threshold': result['val_threshold'],
        'val_recall': result['val_recall'],
        'val_precision': result['val_precision'],
        'run_time_seconds': result.get('run_time_seconds'),
        'ml_train_log_path': os.path.join(output_dir, 'ml_train.tsv'),
    }
