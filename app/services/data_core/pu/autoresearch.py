"""PU Learning autoresearch 演进循环（后台线程）。"""
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flask import Flask

from app.services.data_core.pu.bagging import (
    PU_TRAIN_LOG_FILENAME,
    PUTrainStoppedError,
    PUTrainTimeoutError,
    promote_autoresearch_predictions,
    terminate_active_pu_training,
)
from app.services.data_core.pu.param_optimizer import (
    DEFAULT_PU_PARAMS,
    pu_params_equal,
    suggest_pu_params_autoresearch,
)

MAX_SAME_PARAMS_LLM_RETRIES = int(
    os.environ.get('PU_AUTORESEARCH_MAX_SAME_PARAM_RETRIES', '5')
)
from app.services.data_core.pu.training_runner import execute_pu_model_training


@dataclass
class AutoResearchState:
    running: bool = False
    stop_requested: bool = False
    iteration: int = 0
    max_invalid_iterations: int = 3
    invalid_streak: int = 0
    best_f1: Optional[float] = None
    last_f1: Optional[float] = None
    stop_reason: Optional[str] = None
    last_error: Optional[str] = None
    current_params: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, str]] = field(default_factory=list)
    last_run_result: Optional[Dict[str, Any]] = None


_lock = threading.RLock()
_state = AutoResearchState()
_thread: Optional[threading.Thread] = None


def _append_log(message: str, level: str = 'info') -> None:
    print(f'[autoresearch] {message}')
    with _lock:
        entry = {
            'time': time.strftime('%H:%M:%S'),
            'message': message,
            'type': level,
        }
        _state.logs.append(entry)
        if len(_state.logs) > 500:
            _state.logs = _state.logs[-500:]


def _is_stop_requested() -> bool:
    with _lock:
        return _state.stop_requested


def _interruptible_sleep(seconds: float) -> bool:
    """分段 sleep，便于及时响应停止。返回 True 表示已请求停止。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if _is_stop_requested():
            return True
        time.sleep(min(0.2, deadline - time.monotonic()))
    return _is_stop_requested()


def _pu_train_log_path(project_root: str) -> str:
    return os.path.join(
        project_root, 'data', 'results', 'pu_learning', PU_TRAIN_LOG_FILENAME
    )


def _read_f1_values_from_pu_train(project_root: str) -> List[float]:
    """解析 pu_train.tsv 中所有有效 F1 值（跳过表头、NAN、timeout）。"""
    path = _pu_train_log_path(project_root)
    if not os.path.exists(path):
        return []

    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        return []

    header = lines[0].split('|')
    if 'F1' not in header:
        return []
    f1_idx = header.index('F1')

    values: List[float] = []
    for row_line in lines[1:]:
        row = row_line.split('|')
        if len(row) <= f1_idx:
            continue
        raw = row[f1_idx].strip()
        if raw.upper() == 'NAN' or raw.lower() == 'timeout':
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def read_best_f1_from_pu_train(project_root: str) -> Optional[float]:
    """读取 pu_train.tsv 历史最高 F1。"""
    values = _read_f1_values_from_pu_train(project_root)
    return max(values) if values else None


def read_latest_f1_from_pu_train(project_root: str) -> Optional[float]:
    """读取 pu_train.tsv 最后一行的 F1；NAN/timeout/解析失败返回 None。"""
    values = _read_f1_values_from_pu_train(project_root)
    return values[-1] if values else None


def _update_f1_streak(project_root: str) -> None:
    f1 = read_latest_f1_from_pu_train(project_root)
    with _lock:
        _state.last_f1 = f1
        if f1 is None:
            _state.invalid_streak += 1
            return
        if _state.best_f1 is None or f1 > _state.best_f1:
            _state.best_f1 = f1
            _state.invalid_streak = 0
        else:
            _state.invalid_streak += 1


def _apply_params_to_state(params: Dict[str, Any]) -> None:
    with _lock:
        _state.current_params = dict(params)


def get_autoresearch_status() -> Dict[str, Any]:
    with _lock:
        return {
            'running': _state.running,
            'stopping': _state.running and _state.stop_requested,
            'stop_requested': _state.stop_requested,
            'iteration': _state.iteration,
            'max_invalid_iterations': _state.max_invalid_iterations,
            'invalid_streak': _state.invalid_streak,
            'best_f1': _state.best_f1,
            'last_f1': _state.last_f1,
            'stop_reason': _state.stop_reason,
            'last_error': _state.last_error,
            'current_params': dict(_state.current_params),
            'logs': list(_state.logs),
            'last_run_result': _state.last_run_result,
        }


def request_autoresearch_stop() -> Dict[str, Any]:
    with _lock:
        if not _state.running:
            _append_log('停止请求：autoresearch 当前未在运行', 'warning')
            return {'success': True, 'message': '当前未在运行', 'already_stopped': True}
        if _state.stop_requested:
            _append_log('停止请求：已在停止中，将终止当前 LLM/训练步骤…', 'warning')
        else:
            _state.stop_requested = True
            _append_log('已收到停止按钮，正在终止当前步骤…', 'warning')

    terminated = terminate_active_pu_training()
    if terminated:
        _append_log('已发送终止信号给正在运行的 PU 训练子进程', 'warning')
    else:
        _append_log('当前无运行中的训练子进程（可能处于 LLM 调用阶段，完成后将退出）', 'info')

    return {
        'success': True,
        'message': '停止信号已发送',
        'training_terminated': terminated,
    }


def start_autoresearch(
    app: Flask,
    *,
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_invalid_iterations: int,
    label_col: str,
    dataset_type: str,
    estimated_positive_rate: float,
    initial_params: Dict[str, Any],
    num_boost_round: int,
    timeout_seconds: int,
    log_folder: Optional[str],
) -> Dict[str, Any]:
    global _thread

    project_root = app.config['PROJECT_ROOT']
    historical_best_f1 = read_best_f1_from_pu_train(project_root)

    with _lock:
        if _state.running:
            return {'success': False, 'error': 'autoresearch 已在运行中'}
        _state.running = True
        _state.stop_requested = False
        _state.iteration = 0
        _state.max_invalid_iterations = max(1, int(max_invalid_iterations))
        _state.invalid_streak = 0
        _state.best_f1 = historical_best_f1
        _state.last_f1 = None
        _state.stop_reason = None
        _state.last_error = None
        _state.logs = []
        _state.last_run_result = None
        _state.current_params = dict(initial_params or DEFAULT_PU_PARAMS)

    _append_log(
        f'autoresearch 启动，最大无效迭代次数={_state.max_invalid_iterations}',
        'info',
    )
    if historical_best_f1 is not None:
        _append_log(
            f'历史最优 F1={historical_best_f1:.6g}，'
            f'仅当本轮 F1 严格超过该值时才更新 train/test_predictions.csv',
            'info',
        )
    else:
        _append_log(
            '尚无 pu_train.tsv 历史记录，首次有效训练将写入 predictions CSV',
            'info',
        )

    def _loop():
        with app.app_context():
            project_root = app.config['PROJECT_ROOT']
            output_dir = os.path.join(project_root, 'data', 'results', 'pu_learning')
            try:
                _autoresearch_loop(
                    project_root=project_root,
                    output_dir=output_dir,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    label_col=label_col,
                    dataset_type=dataset_type,
                    estimated_positive_rate=estimated_positive_rate,
                    num_boost_round=num_boost_round,
                    timeout_seconds=timeout_seconds,
                    log_folder=log_folder,
                )
            except Exception as exc:
                _append_log(f'autoresearch 异常终止: {exc}', 'error')
                with _lock:
                    _state.last_error = str(exc)
                    if not _state.stop_reason:
                        _state.stop_reason = 'error'
            finally:
                with _lock:
                    _state.running = False
                    if _state.stop_requested and not _state.stop_reason:
                        _state.stop_reason = 'user_stop'
                _append_log(
                    f'autoresearch 已结束（{_state.stop_reason or "unknown"}）',
                    'success' if _state.stop_reason == 'converged' else 'info',
                )

    _thread = threading.Thread(target=_loop, name='pu-autoresearch', daemon=True)
    _thread.start()
    return {'success': True, 'message': 'autoresearch 已启动'}


def _should_stop() -> Optional[str]:
    with _lock:
        if _state.stop_requested:
            return 'user_stop'
        if _state.invalid_streak >= _state.max_invalid_iterations:
            return 'max_invalid_iterations'
    return None


def _autoresearch_loop(
    *,
    project_root: str,
    output_dir: str,
    api_key: str,
    base_url: Optional[str],
    model: str,
    label_col: str,
    dataset_type: str,
    estimated_positive_rate: float,
    num_boost_round: int,
    timeout_seconds: int,
    log_folder: Optional[str],
) -> None:
    timeout_minutes = max(1, timeout_seconds // 60)
    while True:
        reason = _should_stop()
        if reason:
            with _lock:
                _state.stop_reason = reason
            if reason == 'user_stop':
                _append_log('用户停止 autoresearch', 'warning')
            elif reason == 'max_invalid_iterations':
                _append_log(
                    f'连续 {_state.max_invalid_iterations} 次 F1 未提升，自动停止',
                    'warning',
                )
            break

        with _lock:
            _state.iteration += 1
            iteration = _state.iteration
            params = dict(_state.current_params)
            prev_best_f1 = _state.best_f1
            invalid_streak = _state.invalid_streak

        _append_log(f'—— 第 {iteration} 轮 ——', 'info')
        _append_log('正在调用大模型…', 'info')

        if _should_stop():
            with _lock:
                _state.stop_reason = 'user_stop'
            _append_log('停止信号：跳过 LLM 调用', 'warning')
            break

        baseline_params = dict(params)
        param_retry_feedback: Optional[str] = None
        suggestion: Optional[Dict[str, Any]] = None
        params_resolved = False

        for llm_attempt in range(1, MAX_SAME_PARAMS_LLM_RETRIES + 1):
            if _should_stop():
                with _lock:
                    _state.stop_reason = 'user_stop'
                _append_log('停止信号：跳过 LLM 调用', 'warning')
                break

            try:
                suggestion = suggest_pu_params_autoresearch(
                    project_root=project_root,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    log_folder=log_folder,
                    iteration=iteration,
                    best_f1=prev_best_f1,
                    invalid_streak=invalid_streak,
                    param_retry_feedback=param_retry_feedback,
                )
            except Exception as exc:
                _append_log(f'大模型调用失败: {exc}', 'error')
                with _lock:
                    _state.last_error = str(exc)
                    _state.invalid_streak += 1
                reason = _should_stop()
                if reason:
                    with _lock:
                        _state.stop_reason = reason
                    break
                if _interruptible_sleep(1):
                    with _lock:
                        _state.stop_reason = 'user_stop'
                    break
                suggestion = None
                break

            proposed = suggestion['params']
            if not pu_params_equal(proposed, baseline_params):
                params = proposed
                _apply_params_to_state(params)
                if suggestion.get('reasoning'):
                    _append_log('LLM: ' + suggestion['reasoning'], 'info')
                params_resolved = True
                break

            prev_reasoning = (suggestion.get('reasoning') or '').strip() or '（无）'
            param_retry_feedback = (
                f'检测到错误，下发"params"未更新，'
                f'上一轮获取的"reasoning"：{prev_reasoning}'
            )
            _append_log(
                f'下发参数与本轮基准一致（LLM 尝试 {llm_attempt}/'
                f'{MAX_SAME_PARAMS_LLM_RETRIES}），跳过训练并重新请求 LLM…',
                'warning',
            )

        if _is_stop_requested():
            with _lock:
                _state.stop_reason = 'user_stop'
            break

        if not params_resolved:
            if suggestion is None:
                continue
            _append_log(
                f'连续 {MAX_SAME_PARAMS_LLM_RETRIES} 次返回相同参数，跳过本轮训练',
                'warning',
            )
            with _lock:
                _state.invalid_streak += 1
            reason = _should_stop()
            if reason:
                with _lock:
                    _state.stop_reason = reason
                break
            if _interruptible_sleep(0.5):
                with _lock:
                    _state.stop_reason = 'user_stop'
                break
            continue

        if _should_stop():
            with _lock:
                _state.stop_reason = 'user_stop'
            _append_log('停止信号：跳过本轮训练', 'warning')
            break

        _append_log(
            '开始训练: '
            + ', '.join(f'{k}={params[k]}' for k in sorted(params.keys())[:4])
            + '…',
            'info',
        )

        timed_out = False
        run_error = None
        training_recorded = False
        run_result = None
        try:
            run_result = execute_pu_model_training(
                project_root=project_root,
                label_col=label_col,
                dataset_type=dataset_type,
                estimated_positive_rate=estimated_positive_rate,
                pu_params=params,
                num_boost_round=num_boost_round,
                timeout_seconds=timeout_seconds,
                save_prediction_files=False,
                stop_checker=_is_stop_requested,
            )
            with _lock:
                _state.last_run_result = run_result
            ev = run_result.get('test_evaluation', {})
            _append_log(
                f'训练完成 F1={ev.get("f1", 0):.4f} '
                f'召回={ev.get("recall", 0):.4f} 精准={ev.get("precision", 0):.4f}',
                'success',
            )
            training_recorded = True
        except PUTrainStoppedError as exc:
            run_error = str(exc)
            _append_log('训练已被用户停止', 'warning')
            with _lock:
                _state.stop_reason = 'user_stop'
            break
        except PUTrainTimeoutError as exc:
            timed_out = True
            run_error = str(exc)
            _append_log(
                f'训练超时（{timeout_minutes} 分钟），已记入 pu_train.tsv',
                'error',
            )
            training_recorded = True
        except Exception as exc:
            run_error = str(exc)
            _append_log(f'训练失败: {exc}', 'error')
            _append_log(traceback.format_exc(), 'error')
            with _lock:
                _state.last_error = run_error
                _state.stop_reason = 'error'
            _append_log('发生不可恢复错误，autoresearch 已停止', 'error')
            break

        with _lock:
            if run_error:
                _state.last_error = run_error

        if training_recorded:
            _update_f1_streak(project_root)
        else:
            with _lock:
                _state.invalid_streak += 1
                _state.last_f1 = None

        with _lock:
            streak = _state.invalid_streak
            best = _state.best_f1
            last = _state.last_f1

        run_f1 = None
        if run_result and not timed_out:
            run_f1 = run_result.get('test_evaluation', {}).get('f1')
        elif last is not None:
            run_f1 = last

        improved = (
            run_f1 is not None
            and (prev_best_f1 is None or run_f1 > prev_best_f1)
        )
        if improved:
            promote_autoresearch_predictions(output_dir)
            _append_log(
                f'本轮 F1={run_f1:.4f} 刷新最优，已更新 train/test_predictions.csv',
                'success',
            )
        elif run_f1 is not None:
            best_display = best if best is not None else prev_best_f1
            if best_display is not None:
                _append_log(
                    f'本轮 F1={run_f1:.4f} 未超过最优 {best_display:.4f}，'
                    f'未更新 predictions CSV',
                    'info',
                )

        if timed_out or run_f1 is None:
            _append_log(
                f'本轮无效（超时或无 F1），连续无效 {streak}/{_state.max_invalid_iterations}',
                'warning',
            )
        elif improved:
            _append_log(f'F1 提升: {run_f1:.4f}，最佳 {best:.4f}', 'success')
        else:
            best_display = best if best is not None else prev_best_f1
            _append_log(
                f'F1 未提升: {run_f1:.4f}，最佳 {best_display:.4f}，'
                f'连续无效 {streak}/{_state.max_invalid_iterations}',
                'warning',
            )

        reason = _should_stop()
        if reason:
            with _lock:
                _state.stop_reason = reason
            break

        if _interruptible_sleep(0.5):
            with _lock:
                _state.stop_reason = 'user_stop'
            break
