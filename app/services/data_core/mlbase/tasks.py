"""MLBase 对比与 Test 评估后台任务。"""
from typing import Any, Dict, Optional

from flask import Flask

from app.services.data_core.mlbase.comparison import (
    load_comparison_from_disk,
    run_mlbase_comparison,
)
from app.services.data_core.mlbase.test_eval import (
    load_mlbase_test_metrics,
    run_mlbase_test_evaluation,
)
from app.services.data_core.shared.background_task import BackgroundTaskRunner

_comparison_runner = BackgroundTaskRunner(
    '[mlbase_comparison]',
    max_logs=200,
    thread_name='mlbase_comparison',
)
_test_runner = BackgroundTaskRunner('[mlbase_test]', max_logs=200, thread_name='mlbase_test')


def _comparison_append_log(message: str, level: str = 'info') -> None:
    _comparison_runner.append_log(message, level)


def _enrich_comparison_status(
    payload: Dict[str, Any], project_root: Optional[str]
) -> None:
    if _comparison_runner.result and not _comparison_runner.running and not _comparison_runner.error:
        payload['has_results'] = True
        payload['comparison'] = _comparison_runner.result
    elif (
        project_root
        and not _comparison_runner.running
        and not _comparison_runner.error
        and not payload.get('has_results')
    ):
        disk = load_comparison_from_disk(project_root)
        if disk:
            payload['has_results'] = True
            payload['comparison'] = disk


def get_mlbase_comparison_status(
    log_since: int = 0,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    return _comparison_runner.get_status(
        log_since, project_root, enrich_fn=_enrich_comparison_status
    )


def is_mlbase_comparison_running() -> bool:
    return _comparison_runner.running


def start_mlbase_comparison_task(
    app: Flask,
    *,
    project_root: str,
    run_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    from app.services.data_core.mlbase.autoresearch import is_mlbase_autoresearch_running

    if is_mlbase_autoresearch_running():
        return {'success': False, 'error': 'MLBase autoresearch 正在运行，请稍后再试'}

    def worker() -> None:
        _comparison_append_log('后台 ML 对比已启动', 'info')
        comparison = run_mlbase_comparison(project_root, **run_kwargs)
        _comparison_append_log('ML 对比完成', 'success')
        _comparison_runner.set_result(comparison)

    return _comparison_runner.start(
        app,
        worker,
        busy_message='ML 对比正在运行中，请稍候',
        pre_log='已提交 ML 对比任务',
    )


def _test_append_log(message: str, level: str = 'info') -> None:
    _test_runner.append_log(message, level)


def _enrich_test_status(payload: Dict[str, Any], project_root: Optional[str]) -> None:
    if _test_runner.result and not _test_runner.running and not _test_runner.error:
        payload['has_results'] = True
        payload['metrics'] = _test_runner.result
    elif (
        project_root
        and not _test_runner.running
        and not _test_runner.error
        and not payload.get('has_results')
    ):
        disk = load_mlbase_test_metrics(project_root)
        if disk:
            payload['has_results'] = True
            payload['metrics'] = disk


def get_mlbase_test_status(
    log_since: int = 0,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    return _test_runner.get_status(log_since, project_root, enrich_fn=_enrich_test_status)


def start_mlbase_test_task(
    app: Flask,
    *,
    project_root: str,
    run_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    variant = run_kwargs.get('variant', 'top_features')

    def worker() -> None:
        _test_append_log('后台 Test 评估已启动', 'info')
        metrics = run_mlbase_test_evaluation(project_root, **run_kwargs)
        _test_append_log(
            f'Test 完成: 精度 {metrics["test_precision"]:.4f}, '
            f'召回 {metrics["test_recall"]:.4f}',
            'success',
        )
        _test_runner.set_result(metrics)

    return _test_runner.start(
        app,
        worker,
        busy_message='Test 集评估正在运行中，请稍候',
        pre_log=f'已提交 Test 评估（{variant}）',
    )
