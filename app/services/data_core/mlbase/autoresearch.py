"""MLBase autoresearch 演进循环（后台线程）。"""
import json
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flask import Flask

from app.services.data_core.mlbase.comparison import load_comparison_from_disk
from app.services.data_core.mlbase.param_optimizer import (
    DEFAULT_MLBASE_PARAMS,
    _clamp_mlbase_ai_params,
    _clamp_mlbase_params,
    merge_mlbase_params,
    ml_params_equal,
    suggest_mlbase_params_autoresearch,
)
from app.services.data_core.mlbase.train_log import (
    read_best_precision_from_ml_train,
    read_latest_precision_from_ml_train,
    seed_ml_train_baseline_from_comparison,
)
from app.services.data_core.mlbase.training_runner import (
    MLBaseTrainStoppedError,
    MLBaseTrainTimeoutError,
    execute_mlbase_variant_training,
    terminate_active_mlbase_training,
)

MAX_SAME_PARAMS_LLM_RETRIES = int(
    os.environ.get('ML_AUTORESEARCH_MAX_SAME_PARAM_RETRIES', '5')
)

MLBASE_COMPARISON_JSON = 'mlbase_comparison.json'


@dataclass
class MLBaseAutoResearchState:
    running: bool = False
    stop_requested: bool = False
    iteration: int = 0
    max_invalid_iterations: int = 3
    invalid_streak: int = 0
    best_precision: Optional[float] = None
    last_precision: Optional[float] = None
    stop_reason: Optional[str] = None
    last_error: Optional[str] = None
    variant: str = 'top_features'
    current_params: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, str]] = field(default_factory=list)
    last_run_result: Optional[Dict[str, Any]] = None


_lock = threading.RLock()
_state = MLBaseAutoResearchState()
_thread: Optional[threading.Thread] = None


def _append_log(message: str, level: str = 'info') -> None:
    print(f'[mlbase-autoresearch] {message}')
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


def _output_dir(project_root: str) -> str:
    return os.path.join(project_root, 'data', 'results', 'feature_selection')


def _update_precision_streak(project_root: str) -> None:
    precision = read_latest_precision_from_ml_train(project_root, _output_dir(project_root))
    with _lock:
        _state.last_precision = precision
        if precision is None:
            _state.invalid_streak += 1
            return
        if _state.best_precision is None or precision > _state.best_precision:
            _state.best_precision = precision
            _state.invalid_streak = 0
        else:
            _state.invalid_streak += 1


def _apply_params_to_state(params: Dict[str, Any]) -> None:
    with _lock:
        _state.current_params = _clamp_mlbase_params(params)


def _apply_ai_params_to_state(ai_params: Dict[str, Any]) -> None:
    with _lock:
        _state.current_params = merge_mlbase_params(_state.current_params, ai_params)


def promote_mlbase_autoresearch_result(
    project_root: str,
    variant: str,
    experiment: Dict[str, Any],
) -> str:
    """仅更新 mlbase_comparison.json 中对应 variant 分支。"""
    output_dir = _output_dir(project_root)
    path = os.path.join(output_dir, MLBASE_COMPARISON_JSON)
    comparison = load_comparison_from_disk(project_root) or {
        'full_features': None,
        'top_features': None,
        'warning': None,
        'summary': {},
    }
    comparison[variant] = experiment

    full = comparison.get('full_features')
    top = comparison.get('top_features')
    if full and top:
        comparison['summary'] = {
            'full_val_precision': full.get('val_precision'),
            'top_val_precision': top.get('val_precision'),
            'full_val_recall': full.get('val_recall'),
            'top_val_recall': top.get('val_recall'),
            'precision_delta': top.get('val_precision', 0) - full.get('val_precision', 0),
            'recall_delta': top.get('val_recall', 0) - full.get('val_recall', 0),
            'full_feature_count': full.get('feature_count'),
            'top_feature_count': top.get('feature_count'),
        }
    elif experiment:
        comparison['summary'] = {
            f'{variant}_val_precision': experiment.get('val_precision'),
            f'{variant}_val_recall': experiment.get('val_recall'),
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    return path


def get_mlbase_autoresearch_status() -> Dict[str, Any]:
    with _lock:
        return {
            'running': _state.running,
            'stopping': _state.running and _state.stop_requested,
            'stop_requested': _state.stop_requested,
            'iteration': _state.iteration,
            'max_invalid_iterations': _state.max_invalid_iterations,
            'invalid_streak': _state.invalid_streak,
            'best_precision': _state.best_precision,
            'last_precision': _state.last_precision,
            'variant': _state.variant,
            'stop_reason': _state.stop_reason,
            'last_error': _state.last_error,
            'current_params': dict(_state.current_params),
            'logs': list(_state.logs),
            'last_run_result': _state.last_run_result,
        }


def is_mlbase_autoresearch_running() -> bool:
    with _lock:
        return _state.running


def request_mlbase_autoresearch_stop() -> Dict[str, Any]:
    with _lock:
        if not _state.running:
            _append_log('停止请求：MLBase autoresearch 当前未在运行', 'warning')
            return {'success': True, 'message': '当前未在运行', 'already_stopped': True}
        if _state.stop_requested:
            _append_log('停止请求：已在停止中…', 'warning')
        else:
            _state.stop_requested = True
            _append_log('已收到停止按钮，正在终止当前步骤…', 'warning')

    terminated = terminate_active_mlbase_training()
    if terminated:
        _append_log('已发送终止信号给正在运行的 MLBase 训练子进程', 'warning')
    else:
        _append_log('当前无运行中的训练子进程（可能处于 LLM 调用阶段）', 'info')

    return {
        'success': True,
        'message': '停止信号已发送',
        'training_terminated': terminated,
    }


def start_mlbase_autoresearch(
    app: Flask,
    *,
    api_key: str,
    base_url: Optional[str],
    model: str,
    max_invalid_iterations: int,
    variant: str,
    label_col: str,
    dataset_type: str,
    initial_params: Dict[str, Any],
    timeout_seconds: int,
    log_folder: Optional[str],
) -> Dict[str, Any]:
    global _thread

    if variant not in ('full_features', 'top_features'):
        return {'success': False, 'error': 'variant 须为 full_features 或 top_features'}

    project_root = app.config['PROJECT_ROOT']
    output_dir = _output_dir(project_root)

    seeded_path = seed_ml_train_baseline_from_comparison(
        project_root,
        variant,
        initial_params=initial_params,
    )
    historical_best = read_best_precision_from_ml_train(project_root, output_dir)

    from app.services.data_core.mlbase.tasks import is_mlbase_comparison_running

    if is_mlbase_comparison_running():
        return {'success': False, 'error': 'ML 对比正在运行，请稍后再启动 autoresearch'}

    with _lock:
        if _state.running:
            return {'success': False, 'error': 'MLBase autoresearch 已在运行中'}
        _state.running = True
        _state.stop_requested = False
        _state.iteration = 0
        _state.max_invalid_iterations = max(1, int(max_invalid_iterations))
        _state.invalid_streak = 0
        _state.best_precision = historical_best
        _state.last_precision = None
        _state.stop_reason = None
        _state.last_error = None
        _state.variant = variant
        _state.current_params = _clamp_mlbase_params(initial_params or DEFAULT_MLBASE_PARAMS)
        _state.logs = []
        _state.last_run_result = None

    _append_log(
        f'autoresearch 启动 variant={variant}，'
        f'最大无效迭代={_state.max_invalid_iterations}',
        'info',
    )
    if historical_best is not None:
        _append_log(
            f'历史最优验证精度={historical_best:.4f}，仅当本轮严格超过时才更新 comparison',
            'info',
        )
    else:
        _append_log('尚无 ml_train.tsv 历史，首次有效训练将写入日志并可能更新 comparison', 'info')
    if seeded_path:
        _append_log('已将 ML 对比结果写入 ml_train.tsv 作为 baseline 参考', 'info')

    def _loop() -> None:
        with app.app_context():
            try:
                _autoresearch_loop(
                    project_root=project_root,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    label_col=label_col,
                    dataset_type=dataset_type,
                    variant=variant,
                    timeout_seconds=timeout_seconds,
                    log_folder=log_folder,
                )
            except Exception as exc:
                _append_log(f'autoresearch 异常终止: {exc}', 'error')
                _append_log(traceback.format_exc(), 'error')
                with _lock:
                    _state.last_error = str(exc)
                    _state.stop_reason = 'error'
            finally:
                with _lock:
                    _state.running = False
                    _state.stop_requested = False
                _append_log(
                    f'autoresearch 已结束（{_state.stop_reason or "unknown"}）',
                    'info',
                )

    _thread = threading.Thread(target=_loop, name='mlbase-autoresearch', daemon=True)
    _thread.start()
    return {'success': True, 'message': 'MLBase autoresearch 已启动'}


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
    api_key: str,
    base_url: Optional[str],
    model: str,
    label_col: str,
    dataset_type: str,
    variant: str,
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
                    f'连续 {_state.max_invalid_iterations} 次验证精度未提升，自动停止',
                    'warning',
                )
            break

        with _lock:
            _state.iteration += 1
            iteration = _state.iteration
            params = dict(_state.current_params)
            prev_best = _state.best_precision
            invalid_streak = _state.invalid_streak

        _append_log(f'—— 第 {iteration} 轮 ——', 'info')
        _append_log('正在调用大模型…', 'info')

        if _should_stop():
            with _lock:
                _state.stop_reason = 'user_stop'
            _append_log('停止信号：跳过 LLM 调用', 'warning')
            break

        baseline_params = _clamp_mlbase_ai_params(params)
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
                suggestion = suggest_mlbase_params_autoresearch(
                    project_root=project_root,
                    api_key=api_key,
                    variant=variant,
                    label_col=label_col,
                    dataset_type=dataset_type,
                    base_url=base_url,
                    model=model,
                    log_folder=log_folder,
                    iteration=iteration,
                    best_precision=prev_best,
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
            if not ml_params_equal(proposed, baseline_params):
                params = merge_mlbase_params(params, proposed)
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
            run_result = execute_mlbase_variant_training(
                project_root=project_root,
                variant=variant,
                label_col=label_col,
                dataset_type=dataset_type,
                ml_params=params,
                timeout_seconds=timeout_seconds,
                stop_checker=_is_stop_requested,
            )
            with _lock:
                _state.last_run_result = run_result
            _append_log(
                f'训练完成 精度={run_result.get("val_precision", 0):.4f} '
                f'召回={run_result.get("val_recall", 0):.4f} '
                f'阈值={run_result.get("val_threshold", 0):.4f}',
                'success',
            )
            training_recorded = True
        except MLBaseTrainStoppedError:
            _append_log('训练已被用户停止', 'warning')
            with _lock:
                _state.stop_reason = 'user_stop'
            break
        except MLBaseTrainTimeoutError as exc:
            timed_out = True
            run_error = str(exc)
            _append_log(
                f'训练超时（{timeout_minutes} 分钟），已记入 ml_train.tsv',
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
            _update_precision_streak(project_root)
        else:
            with _lock:
                _state.invalid_streak += 1
                _state.last_precision = None

        with _lock:
            streak = _state.invalid_streak
            best = _state.best_precision
            last = _state.last_precision

        run_precision = None
        experiment = None
        if run_result and not timed_out:
            run_precision = run_result.get('val_precision')
            experiment = run_result.get('experiment')
        elif last is not None:
            run_precision = last

        improved = (
            run_precision is not None
            and (prev_best is None or run_precision > prev_best)
        )
        if improved and experiment:
            promote_mlbase_autoresearch_result(project_root, variant, experiment)
            _append_log(
                f'本轮精度={run_precision:.4f} 刷新最优，已更新 mlbase_comparison.json',
                'success',
            )
        elif run_precision is not None:
            best_display = best if best is not None else prev_best
            if best_display is not None:
                _append_log(
                    f'本轮精度={run_precision:.4f} 未超过最优 {best_display:.4f}，'
                    f'未更新 comparison',
                    'info',
                )

        if timed_out or run_precision is None:
            _append_log(
                f'本轮无效（超时或无精度），连续无效 {streak}/{_state.max_invalid_iterations}',
                'warning',
            )
        elif improved:
            _append_log(f'精度提升: {run_precision:.4f}，最佳 {best:.4f}', 'success')
        else:
            best_display = best if best is not None else prev_best
            _append_log(
                f'精度未提升: {run_precision:.4f}，最佳 {best_display:.4f}，'
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
