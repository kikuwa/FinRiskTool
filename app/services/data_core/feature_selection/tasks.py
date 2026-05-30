"""特征选择与 RFECV 后台任务。"""
import json
import os
import traceback
from typing import Any, Dict, Optional

from flask import Flask

from app.services.data_core.feature_selection.pipeline import (
    FEATURE_CANDIDATES_CSV,
    FEATURE_META_JSON,
    run_feature_selection_pipeline,
)
from app.services.data_core.feature_selection.rfecv import run_rfecv_reselection
from app.services.data_core.shared.background_task import BackgroundTaskRunner
from app.services.data_core.shared.data_loader import DataLoader

_fs_runner = BackgroundTaskRunner(
    '[feature_selection]',
    max_logs=300,
    thread_name='feature_selection',
)
_rfecv_runner = BackgroundTaskRunner('[rfecv]', max_logs=200, thread_name='rfecv_reselection')
_label_col: Optional[str] = None
_dataset_type: Optional[str] = None


def _fs_append_log(message: str, level: str = 'info') -> None:
    _fs_runner.append_log(message, level)


def _disk_feature_selection_summary(project_root: str) -> Optional[Dict[str, Any]]:
    output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
    cand_path = os.path.join(output_dir, FEATURE_CANDIDATES_CSV)
    if not os.path.isfile(cand_path):
        return None
    meta = {}
    meta_path = os.path.join(output_dir, FEATURE_META_JSON)
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            pass
    try:
        import pandas as pd
        n = len(pd.read_csv(cand_path, usecols=[0]))
    except Exception:
        with open(cand_path, 'r', encoding='utf-8') as f:
            n = max(0, sum(1 for _ in f) - 1)
    return {
        'has_results': True,
        'candidates_on_disk': True,
        'candidate_count': n,
        'meta': meta,
    }


def _enrich_feature_selection_status(
    payload: Dict[str, Any], project_root: Optional[str]
) -> None:
    payload['label_col'] = _label_col
    payload['dataset_type'] = _dataset_type
    if _fs_runner.result and not _fs_runner.running and not _fs_runner.error:
        payload['has_results'] = True
        payload['meta'] = _fs_runner.result.get('meta')
        payload['candidate_count'] = len(_fs_runner.result.get('candidates', []))
        payload['default_selected'] = (_fs_runner.result.get('default_selected') or [])[:10]
    elif (
        project_root
        and not _fs_runner.running
        and not _fs_runner.error
        and not payload.get('has_results')
    ):
        disk = _disk_feature_selection_summary(project_root)
        if disk:
            payload.update(disk)


def get_feature_selection_status(
    log_since: int = 0,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    return _fs_runner.get_status(
        log_since, project_root, enrich_fn=_enrich_feature_selection_status
    )


def start_feature_selection_task(
    app: Flask,
    *,
    project_root: str,
    label_col: str,
    dataset_type: str,
    estimated_positive_rate: float,
    fe_params: Optional[dict] = None,
) -> Dict[str, Any]:
    global _label_col, _dataset_type

    upload_dir = os.path.join(project_root, 'data', 'uploads')
    output_dir = os.path.join(project_root, 'data', 'results', 'feature_selection')
    pu_pred_path = os.path.join(
        project_root, 'data', 'results', 'pu_learning', 'train_predictions.csv'
    )

    def worker() -> None:
        try:
            _fs_append_log('后台任务已启动', 'info')
            loader = DataLoader(label_col=label_col)
            if dataset_type == 'full':
                file_path = os.path.join(upload_dir, 'full_dataset.csv')
                if not os.path.exists(file_path):
                    default_path = os.path.join(project_root, 'data', 'train.csv')
                    file_path = default_path if os.path.exists(default_path) else file_path
                if not os.path.exists(file_path):
                    raise FileNotFoundError('未找到数据集，请先上传')
                train_df = loader._load_csv(file_path)
            else:
                train_path = os.path.join(upload_dir, 'train_dataset.csv')
                if not os.path.exists(train_path):
                    raise FileNotFoundError('未找到训练集，请先上传')
                _fs_append_log('正在加载训练集…', 'info')
                train_df = loader._load_csv(train_path)

            loader.validate_data(train_df)
            _fs_append_log(
                f'数据就绪: {len(train_df)} 行，标签列 {label_col}，'
                f'预估正样本比例 {estimated_positive_rate:.2%}',
                'info',
            )
            _fs_append_log('PN + MI + LGB 稳定性运行中（约数分钟）…', 'info')

            result = run_feature_selection_pipeline(
                train_df=train_df,
                label_col=label_col,
                output_dir=output_dir,
                estimated_positive_rate=estimated_positive_rate,
                train_predictions_path=pu_pred_path,
                project_root=project_root,
                fe_params=fe_params,
            )
            n_cand = len(result.get('candidates', []))
            _fs_append_log(f'完成，候选特征 {n_cand} 个', 'success')
            _fs_runner.set_result(result)
        except Exception as exc:
            msg = str(exc)
            _fs_append_log(f'失败: {msg}', 'error')
            print(traceback.format_exc())
            _fs_runner.set_error(msg)

    _label_col = label_col
    _dataset_type = dataset_type
    return _fs_runner.start(
        app,
        worker,
        busy_message='特征选择正在运行中，请稍候',
        pre_log='已提交特征选择任务',
    )


def _rfecv_append_log(message: str, level: str = 'info') -> None:
    _rfecv_runner.append_log(message, level)


def _enrich_rfecv_status(payload: Dict[str, Any], project_root: Optional[str]) -> None:
    if _rfecv_runner.result and not _rfecv_runner.running and not _rfecv_runner.error:
        payload['has_results'] = True
        payload['result'] = _rfecv_runner.result


def get_rfecv_status(
    log_since: int = 0,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    return _rfecv_runner.get_status(log_since, project_root, enrich_fn=_enrich_rfecv_status)


def start_rfecv_task(
    app: Flask,
    *,
    project_root: str,
    label_col: str,
    dataset_type: str,
    output_dir: str,
) -> Dict[str, Any]:
    def worker() -> None:
        result = run_rfecv_reselection(
            project_root,
            label_col=label_col,
            dataset_type=dataset_type,
            output_dir=output_dir,
            log_fn=_rfecv_append_log,
        )
        _rfecv_runner.set_result(result)

    return _rfecv_runner.start(
        app,
        worker,
        busy_message='RFECV 正在运行中，请稍候',
        pre_log='已提交 RFECV 任务',
    )
